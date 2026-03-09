// Build script for the flash-attention CUDA kernels.
// Set FLASH_ATTN_BUILD_DIR to reuse compiled artifacts across cargo builds.
use anyhow::{Context, Result};
use cudaforge::{detect_compute_cap, CudaToolkit, KernelBuilder};
use std::fs;
use std::path::{Path, PathBuf};

const CUTLASS_COMMIT: &str = "57e3cfb47a2d9e0d46eb6335c3dc411498efa198";

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
        kernel_files.extend_from_slice(&["kernels/flash_api_sm90.cu".to_string()]);
    } else {
        kernel_files.extend_from_slice(&["kernels/flash_api_sm80.cu".to_string()]);
    }

    let inst_dir = Path::new("kernels/instantiations");
    let sm_filter = if use_v3 { "_sm90.cu" } else { "_sm80.cu" };
    for entry in fs::read_dir(inst_dir).context("read kernels/instantiations")? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("cu") {
            continue;
        }

        let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if !file_name.ends_with(sm_filter) {
            continue;
        }

        let is_split = file_name.contains("_split_");
        let is_hdim64 = file_name.contains("hdim64");
        let is_hdim128 = file_name.contains("hdim128");
        let is_hdim256 = file_name.contains("hdim256");
        let is_hdim512 = file_name.contains("_512_");
        let is_fp8 = file_name.contains("_e4m3_");

        if flash_context_enabled {
            if !(is_hdim64 || is_hdim128 || is_hdim256) {
                continue;
            }
            if is_fp8 || is_hdim512 {
                continue;
            }
        } else if !flash_decoding_enabled && is_split {
            continue;
        }

        kernel_files.push(path.to_string_lossy().to_string());
    }

    kernel_files.sort();
    Ok(kernel_files)
}

fn header_watch_paths() -> &'static [&'static str] {
    &[
        "kernels/flash.h",
        "kernels/flash_api_impl.h",
        "kernels/flash_fwd_launch_template.h",
        "kernels/flash_fwd_combine_launch_template.h",
        "kernels/flash_fwd_kernel_sm80.h",
        "kernels/flash_fwd_kernel_sm90.h",
        "kernels/tile_size.h",
        "kernels/heuristics.h",
        "kernels/utils.h",
        "kernels/static_switch.h",
        "kernels/cuda_check.h",
    ]
}

fn build_dir() -> Result<PathBuf> {
    let build_dir = match std::env::var("FLASH_ATTN_BUILD_DIR") {
        Ok(path) => PathBuf::from(path),
        Err(_) => {
            let profile = std::env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
            PathBuf::from("target")
                .join(profile)
                .join("build")
                .join("flashattn_build")
        }
    };

    fs::create_dir_all(&build_dir).context("create flashattn build dir")?;
    build_dir
        .canonicalize()
        .context("canonicalize flashattn build dir")
}

fn detect_build_compute_cap() -> usize {
    match detect_compute_cap() {
        Ok(arch) => arch.base,
        Err(err) => {
            println!(
                "cargo:warning=Failed to detect compute capability ({err}); defaulting to sm_90"
            );
            90
        }
    }
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=FLASH_ATTN_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=NVCC_THREADS");

    let toolkit = CudaToolkit::detect().context("detect CUDA toolkit")?;
    if std::env::var("CUDA_INCLUDE_DIR").is_err() {
        println!(
            "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
            toolkit.include_dir.display()
        );
    }

    let compute_cap = detect_build_compute_cap();
    let flash_decoding_enabled = std::env::var("CARGO_FEATURE_FLASH_DECODING").is_ok();
    let flash_context_enabled = std::env::var("CARGO_FEATURE_FLASH_CONTEXT").is_ok();
    let disable_fp8 = compute_cap < 90 || flash_context_enabled;
    let disable_hdim512 = flash_context_enabled;
    let disable_flash_v2 = (90..=100).contains(&compute_cap);
    let disable_flash_v3 = compute_cap < 90 || compute_cap >= 120;

    if disable_flash_v2 && disable_flash_v3 {
        panic!("No flash attention kernels suitable for this arch {compute_cap}");
    }

    let kernel_files = collect_kernel_files(
        flash_decoding_enabled,
        flash_context_enabled,
        !disable_flash_v3,
    )?;
    let build_dir = build_dir()?;
    let out_file = build_dir.join("libflashattention.a");

    let mut builder = KernelBuilder::new()
        .out_dir(build_dir.join("objects"))
        .source_files(&kernel_files)
        .watch(header_watch_paths())
        .include_path("kernels")
        .with_cutlass(Some(CUTLASS_COMMIT))
        .thread_percentage(0.5)
        .max_threads(32)
        .args([
            "-DNDEBUG",
            "-O3",
            "-std=c++17",
            "-Xcompiler",
            "-fPIC",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DCUTLASS_VERSIONS_GENERATED",
            "-DCUTLASS_TEST_LEVEL=0",
            "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-lineinfo",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-DCUTLASS_ENABLE_GDC_FOR_SM90",
            "-Xcompiler=-Wconversion",
            "-Xcompiler=-fno-strict-aliasing",
            "-Xfatbin",
            "-compress-all",
        ]);

    if flash_context_enabled {
        builder = builder.args([
            "-DFLASHATTENTION_DISABLE_HDIM96",
            "-DFLASHATTENTION_DISABLE_HDIM192",
            "-DFLASHATTENTION_DISABLE_HDIMDIFF96",
            "-DFLASHATTENTION_DISABLE_HDIMDIFF192",
        ]);
    }
    if disable_fp8 {
        builder = builder.arg("-DFLASHATTENTION_DISABLE_FP8");
    }
    if disable_hdim512 {
        builder = builder.arg("-DFLASHATTENTION_DISABLE_HDIM512");
    }
    if disable_flash_v2 {
        builder = builder.arg("-DFLASHATTENTION_DISABLE_SM80");
    }
    if disable_flash_v3 {
        builder = builder.arg("-DFLASHATTENTION_DISABLE_SM90");
    }

    builder.build_lib(&out_file)?;

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=flashattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}
