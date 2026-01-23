// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

const CUTLASS_REPO: &str = "https://github.com/NVIDIA/cutlass.git";
const CUTLASS_COMMIT: &str = "7127592069c2fe01b041e174ba4345ef9b279671";
const DEFAULT_NVCC: &str = "/usr/local/cuda/bin/nvcc";

fn find_nvcc() -> Result<PathBuf> {
    if let Ok(nvcc) = std::env::var("NVCC") {
        return Ok(PathBuf::from(nvcc));
    }
    if let Ok(path) = which::which("nvcc") {
        return Ok(path);
    }
    let fallback = PathBuf::from(DEFAULT_NVCC);
    if fallback.exists() {
        return Ok(fallback);
    }
    bail!("nvcc not found; set NVCC or ensure it is on PATH")
}

fn compute_cap() -> Result<usize> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv"])
        .output();

    if let Ok(out) = out {
        let output = String::from_utf8(out.stdout).context("nvidia-smi output was not utf8")?;
        let line = output
            .lines()
            .nth(1)
            .ok_or_else(|| anyhow!("unexpected nvidia-smi output:\n{output}"))?;
        let cap = line
            .trim()
            .parse::<f32>()
            .context("failed to parse compute_cap")?;
        return Ok((cap * 10.0) as usize);
    }

    if let Ok(var) = std::env::var("CUDA_COMPUTE_CAP") {
        let v = var
            .parse::<usize>()
            .context("CUDA_COMPUTE_CAP must be an integer")?;
        return Ok(if v >= 100 { v } else { v * 10 });
    }

    Err(anyhow!(
        "failed to run nvidia-smi; set CUDA_COMPUTE_CAP env var instead"
    ))
}

fn cargo_git_checkouts_dir() -> Result<PathBuf> {
    if let Ok(ch) = std::env::var("CARGO_HOME") {
        return Ok(PathBuf::from(ch).join("git").join("checkouts"));
    }
    let home = std::env::var("HOME").context("HOME not set; set CARGO_HOME or HOME")?;
    Ok(PathBuf::from(home)
        .join(".cargo")
        .join("git")
        .join("checkouts"))
}

fn pick_cutlass_root_dir() -> Result<PathBuf> {
    let cargo_dir = cargo_git_checkouts_dir()?;
    if cargo_dir.exists() {
        return Ok(cargo_dir);
    }
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let out_dir = PathBuf::from(out_dir);
        if out_dir.exists() {
            return Ok(out_dir);
        }
    }
    Ok(PathBuf::from("."))
}

/// Fetch cutlass headers if not already present at the specified commit.
///
/// The headers are cloned to `out_dir/cutlass` using sparse checkout to only
/// fetch the `include/` and `tools/util/include` directories, minimizing download size.
fn fetch_cutlass(out_dir: &PathBuf, commit: &str) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).context("create cutlass output dir")?;
    let cutlass_dir = out_dir.join("cutlass");

    if cutlass_dir.join("include").exists() {
        let output = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&cutlass_dir)
            .output();

        if let Ok(output) = output {
            let current_commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if current_commit == commit {
                return Ok(cutlass_dir);
            }
        }
    }

    if !cutlass_dir.exists() {
        println!("cargo:warning=Cloning cutlass from {}", CUTLASS_REPO);
        let status = Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                CUTLASS_REPO,
                cutlass_dir
                    .to_str()
                    .context("cutlass path contains invalid UTF-8")?,
            ])
            .status()
            .context("Failed to clone cutlass repository")?;

        if !status.success() {
            bail!("git clone failed with status: {}", status);
        }

        let status = Command::new("git")
            .args(["sparse-checkout", "set", "include", "tools/util/include"])
            .current_dir(&cutlass_dir)
            .status()
            .context("Failed to set sparse checkout for cutlass")?;

        if !status.success() {
            bail!("git sparse-checkout failed with status: {}", status);
        }
    } else if !cutlass_dir.join("include").exists() {
        let status = Command::new("git")
            .args(["sparse-checkout", "set", "include", "tools/util/include"])
            .current_dir(&cutlass_dir)
            .status()
            .context("Failed to set sparse checkout for cutlass")?;

        if !status.success() {
            bail!("git sparse-checkout failed with status: {}", status);
        }
    }

    println!("cargo:warning=Checking out cutlass commit {}", commit);
    let status = Command::new("git")
        .args(["fetch", "origin", commit])
        .current_dir(&cutlass_dir)
        .status()
        .context("Failed to fetch cutlass commit")?;

    if !status.success() {
        bail!("git fetch failed with status: {}", status);
    }

    let status = Command::new("git")
        .args(["checkout", commit])
        .current_dir(&cutlass_dir)
        .status()
        .context("Failed to checkout cutlass commit")?;

    if !status.success() {
        bail!("git checkout failed with status: {}", status);
    }

    Ok(cutlass_dir)
}

fn collect_kernel_files(
    flash_decoding_enabled: bool,
    flash_context_enabled: bool,
    use_v3: bool,
) -> Result<Vec<String>> {
    let mut kernel_files = vec![
        "kernels/flash_api_dispatch.cu".to_string(),
        "kernels/flash_fwd_combine.cu".to_string(),
        "kernels/flash_prepare_scheduler.cu".to_string(),
    ];

    if use_v3 {
        kernel_files.extend_from_slice(&[
            "kernels/flash_api_sm90.cu".to_string(),
            "kernels/flash_api_sm90_softcap.cu".to_string(),
        ]);
    } else {
        kernel_files.extend_from_slice(&[
            "kernels/flash_api_sm80.cu".to_string(),
            "kernels/flash_api_sm80_softcap.cu".to_string(),
        ]);
    }

    let inst_dir = Path::new("kernels/instantiations");
    let sm_filter = if use_v3 { "_sm90.cu" } else { "_sm80.cu" };
    for entry in fs::read_dir(inst_dir).context("read kernels/instantiations")? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("cu") {
            let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !file_name.ends_with(sm_filter) {
                continue;
            }
            let is_split = file_name.contains("_split_");
            let is_hdim64 = file_name.contains("hdim64");
            let is_hdim128 = file_name.contains("hdim128");
            let is_packgqa = file_name.contains("packgqa");

            if flash_context_enabled {
                if !(is_hdim64 || is_hdim128) {
                    continue;
                }
                if is_packgqa {
                    continue;
                }
                let is_hdimdiff =
                    file_name.contains("hdimdiff") || has_hdim_diff_pattern(file_name);
                if is_hdimdiff {
                    continue;
                }
                // Context builds include split kernels for 64/128.
            } else if !flash_decoding_enabled && is_split {
                continue;
            }

            kernel_files.push(path.to_string_lossy().to_string());
        }
    }

    kernel_files.sort();
    Ok(kernel_files)
}

fn has_hdim_diff_pattern(file_name: &str) -> bool {
    let mut rest = file_name;
    while let Some(pos) = rest.find("hdim") {
        rest = &rest[pos + 4..];
        let (first_digits, after_first) = take_digits(rest);
        if first_digits.is_empty() {
            continue;
        }
        let after_first = after_first.strip_prefix('_');
        if after_first.is_none() {
            continue;
        }
        let (second_digits, _) = take_digits(after_first.unwrap());
        if !second_digits.is_empty() {
            return true;
        }
    }
    false
}

fn take_digits(s: &str) -> (&str, &str) {
    let mut end = 0;
    for (idx, ch) in s.char_indices() {
        if ch.is_ascii_digit() {
            end = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    s.split_at(end)
}

fn main() -> Result<()> {
    let nvcc_path = find_nvcc()?;
    let cutlass_root = pick_cutlass_root_dir()?;
    let cutlass_dir = fetch_cutlass(&cutlass_root, CUTLASS_COMMIT)?;

    if std::env::var("CUDA_INCLUDE_DIR").is_err() {
        if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
            let cuda_include = PathBuf::from(cuda_home).join("include");
            println!(
                "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
                cuda_include.display()
            );
        } else if Path::new("/usr/local/cuda/include").exists() {
            println!("cargo:rustc-env=CUDA_INCLUDE_DIR=/usr/local/cuda/include");
        }
    }
    let compute_cap = compute_cap().unwrap_or(90);

    let flash_decoding_enabled = std::env::var("CARGO_FEATURE_FLASH_DECODING").is_ok();
    let flash_context_enabled = std::env::var("CARGO_FEATURE_FLASH_CONTEXT").is_ok();
    let disable_fp8 = compute_cap < 90; // no hardware fp8 for sm_70, sm_80
    let disable_flash_v2 = compute_cap >= 90 && compute_cap <= 100;
    let disable_flash_v3 = compute_cap < 90 || compute_cap >= 120; // v3 has poor compatibility with Blackwell

    if disable_flash_v2 && disable_flash_v3 {
        panic!(
            "No flash attention kernels suitable for this arch {}",
            compute_cap
        );
    }
    let use_v3 = !disable_flash_v3;

    let kernel_files = collect_kernel_files(flash_decoding_enabled, flash_context_enabled, use_v3)?;

    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in &kernel_files {
        println!("cargo:rerun-if-changed={}", kernel_file);
    }
    println!("cargo:rerun-if-changed=kernels/flash.h");
    println!("cargo:rerun-if-changed=kernels/flash_api_impl.h");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_launch_template.h");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_combine_launch_template.h");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_kernel_sm80.h");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_kernel_sm90.h");
    println!("cargo:rerun-if-changed=kernels/tile_size.h");
    println!("cargo:rerun-if-changed=kernels/heuristics.h");
    println!("cargo:rerun-if-changed=kernels/utils.h");
    println!("cargo:rerun-if-changed=kernels/static_switch.h");
    println!("cargo:rerun-if-changed=kernels/cuda_check.h");

    let build_dir = match std::env::var("FLASH_ATTN_BUILD_DIR") {
        Err(_) => {
            let profile = std::env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
            PathBuf::from("target")
                .join(profile)
                .join("build")
                .join("flashattn_build")
        }
        Ok(build_dir) => PathBuf::from(build_dir),
    };
    fs::create_dir_all(&build_dir).context("create flashattn build dir")?;
    let build_dir = build_dir
        .canonicalize()
        .context("canonicalize flashattn build dir")?;

    let cutlass_dir_str = cutlass_dir.display();
    let include_root: &'static str = Box::leak(format!("-I{cutlass_dir_str}").into_boxed_str());
    let include_main: &'static str =
        Box::leak(format!("-I{cutlass_dir_str}/include").into_boxed_str());
    let include_tools: &'static str =
        Box::leak(format!("-I{cutlass_dir_str}/tools/util/include").into_boxed_str());
    println!("cargo:warning=Cutlass local folder {}", include_root);
    println!("cargo:warning=Cutlass include folder {}", include_main);
    println!(
        "cargo:warning=Cutlass tools include folder {}",
        include_tools
    );

    let obj_dir = build_dir.join("objects");
    fs::create_dir_all(&obj_dir).context("create object output dir")?;

    let out_file = build_dir.join("libflashattention.a");
    let out_modified = out_file
        .metadata()
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);

    let mut obj_files = Vec::new();
    let mut compile_jobs = Vec::new();

    for input in &kernel_files {
        let input_path = PathBuf::from(input);
        let file_name = input_path.file_name().context("kernel file without name")?;
        let mut obj_path = obj_dir.join(file_name);
        obj_path.set_extension("o");

        let obj_modified = obj_path
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let input_modified = input_path
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let should_compile =
            !obj_path.exists() || input_modified.duration_since(obj_modified).is_ok();

        if should_compile {
            compile_jobs.push((input_path.clone(), obj_path.clone()));
        }
        obj_files.push(obj_path);
    }

    let max_threads = std::cmp::min(
        48,
        (std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(48) as f32
            * 0.5) as usize,
    );
    ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build_global()
        .context("initialize rayon thread pool")?;

    println!(
        "cargo:warning=Building with maximum {} threads",
        max_threads
    );
    let nvcc_threads = match std::env::var("NVCC_THREADS") {
        Ok(value) => {
            let parsed = value
                .parse::<usize>()
                .context("Failed to parse NVCC_THREADS")?;
            if parsed == 0 {
                bail!("NVCC_THREADS must be >= 1");
            }
            Some(parsed)
        }
        Err(_) => Some(2),
    };

    let rebuilt_any = AtomicBool::new(false);
    let target = std::env::var("TARGET").ok();
    compile_jobs
        .par_iter()
        .try_for_each(|(input_path, obj_path)| -> Result<()> {
            let mut command = Command::new(&nvcc_path);
            let gpu_arch = if compute_cap >= 121 {
                "sm_121a".to_string()
            } else if compute_cap >= 120 {
                "sm_120a".to_string()
            } else if compute_cap >= 100 {
                "sm_100a".to_string()
            } else if compute_cap == 90 {
                "sm_90a".to_string()
            } else {
                format!("sm_{}", compute_cap)
            };
            command
                .arg("-O3")
                .arg("-std=c++17")
                .arg(format!("--gpu-architecture={}", gpu_arch))
                .arg("-c")
                .arg("-o")
                .arg(obj_path)
                .arg("-U__CUDA_NO_HALF_OPERATORS__")
                .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
                .arg("-U__CUDA_NO_HALF2_OPERATORS__")
                .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
                .arg("-Ikernels")
                .arg("-DUSE_CUTLASS")
                .arg(include_root)
                .arg(include_main)
                .arg(include_tools)
                .arg("--expt-relaxed-constexpr")
                .arg("--expt-extended-lambda")
                .arg("--use_fast_math")
                .arg("-lineinfo")
                .arg("-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED")
                .arg("-DCUTLASS_ENABLE_GDC_FOR_SM90")
                .arg("-Xfatbin")
                .arg("-compress-all")
                .arg("-Xcompiler")
                .arg("-fPIC");

            if let Some(threads) = nvcc_threads {
                let input_str = input_path.to_string_lossy();
                if input_str.contains("flash_api") {
                    command.arg(format!("--threads={}", threads));
                }
            }

            if flash_context_enabled {
                command
                    .arg("-DFLASHATTENTION_DISABLE_HDIM96")
                    .arg("-DFLASHATTENTION_DISABLE_HDIM192")
                    .arg("-DFLASHATTENTION_DISABLE_HDIM256")
                    .arg("-DFLASHATTENTION_DISABLE_HDIMDIFF64")
                    .arg("-DFLASHATTENTION_DISABLE_HDIMDIFF192");
            }

            if disable_fp8 {
                command.arg("-DFLASHATTENTION_DISABLE_FP8");
            }
            if disable_flash_v2 {
                command.arg("-DFLASHATTENTION_DISABLE_SM80");
            }
            if disable_flash_v3 {
                command.arg("-DFLASHATTENTION_DISABLE_SM90");
            }
            if let Some(target) = target.as_ref() {
                if target.contains("msvc") {
                    command.arg("-D_USE_MATH_DEFINES");
                }
            }

            command.arg(input_path);

            let output = command
                .output()
                .with_context(|| format!("Failed to invoke nvcc for {input_path:?}"))?;
            if !output.status.success() {
                bail!(
                    "nvcc error:\nCommand: {:?}\nstdout:\n{}\nstderr:\n{}",
                    command,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }

            rebuilt_any.store(true, Ordering::Relaxed);
            Ok(())
        })?;

    let out_is_stale = !out_file.exists()
        || rebuilt_any.load(Ordering::Relaxed)
        || obj_files.iter().any(|obj| {
            let obj_modified = obj
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            obj_modified.duration_since(out_modified).is_ok()
        });

    if out_is_stale {
        let mut command = Command::new(&nvcc_path);
        command.arg("--lib");
        command.arg("-o").arg(&out_file);
        command.args(&obj_files);
        let output = command
            .output()
            .context("Failed to invoke nvcc for archive step")?;
        if !output.status.success() {
            bail!(
                "nvcc error (archiving):\nCommand: {:?}\nstdout:\n{}\nstderr:\n{}",
                command,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    } else {
        println!("cargo:warning=Skipping CUDA archive (up-to-date)");
    }

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=flashattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}
