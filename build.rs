//! This build script emits the openblas linking directive if requested

#[cfg(any(feature = "intel-mkl", feature = "cblas"))]
#[derive(PartialEq, Eq)]
enum Library {
    Static,
    Dynamic,
}

pub const SEQUENTIAL: bool = false;
pub const THREADED: bool = !SEQUENTIAL;

pub const LD_DIR: &str = if cfg!(windows) {
    "PATH"
} else if cfg!(target_os = "linux") {
    "LD_LIBRARY_PATH"
} else if cfg!(target_os = "macos") {
    "DYLD_LIBRARY_PATH"
} else {
    ""
};

pub const DEFAULT_ONEAPI_ROOT: &str = if cfg!(windows) {
    "C:/Program Files (x86)/Intel/oneAPI/"
} else {
    "/opt/intel/oneapi/"
};

pub const MKL_CORE: &str = "mkl_core";
pub const MKL_THREAD: &str = if SEQUENTIAL {
    "mkl_sequential"
} else {
    "mkl_intel_thread"
};
pub const THREADING_LIB: &str = if cfg!(windows) { "libiomp5md" } else { "iomp5" };
pub const MKL_INTERFACE: &str = if cfg!(target_pointer_width = "32") {
    "mkl_intel_ilp64"
} else {
    "mkl_intel_lp64"
};

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
pub const UNSUPPORTED_OS_ERROR: _ = "Target OS is not supported. Please contact me";

#[cfg(all(target_os = "windows", target_pointer_width = "32"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/windows/compiler/lib/ia32_win",
    "mkl/latest/lib/ia32",
];

#[cfg(all(target_os = "windows", target_pointer_width = "64"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/windows/compiler/lib/intel64_win",
    "mkl/latest/lib/intel64",
];

#[cfg(all(target_os = "windows", target_pointer_width = "32"))]
pub const SHARED_LIB_DIRS: &[&str] = &[
    "compiler/latest/windows/redist/ia32_win/compiler",
    "mkl/latest/redist/ia32",
];

#[cfg(all(target_os = "windows", target_pointer_width = "64"))]
pub const SHARED_LIB_DIRS: &[&str] = &[
    "compiler/latest/windows/redist/intel64_win/compiler",
    "mkl/latest/redist/intel64",
];

#[cfg(all(target_os = "linux", target_pointer_width = "32"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/linux/compiler/lib/ia32_lin",
    "mkl/latest/lib/ia32",
];

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/linux/compiler/lib/intel64_lin",
    "mkl/latest/lib/intel64",
];

#[cfg(target_os = "linux")]
pub const SHARED_LIB_DIRS: &[&str] = LINK_DIRS;

#[cfg(target_os = "macos")]
const MACOS_COMPILER_PATH: &str = "compiler/latest/mac/compiler/lib";

#[cfg(target_os = "macos")]
pub const LINK_DIRS: &[&str] = &[MACOS_COMPILER_PATH, "mkl/latest/lib"];

#[cfg(target_os = "macos")]
pub const SHARED_LIB_DIRS: &[&str] = &["mkl/latest/lib"];

#[derive(Debug)]
pub enum BuildError {
    OneAPINotFound(std::path::PathBuf),
    OneAPINotADir(std::path::PathBuf),
    PathNotFound(std::env::VarError),
    AddSharedLibDirToPath(String),
}

#[cfg(feature = "intel-mkl")]
fn suggest_setvars_cmd(root: &str) -> String {
    if cfg!(windows) {
        format!("{root}/setvars.bat")
    } else {
        format!("source {root}/setvars.sh")
    }
}

fn main() -> Result<(), BuildError> {
    Ok(())
}
