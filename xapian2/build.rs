use std::path::{Path, PathBuf};

fn main() {
    // Locate a Xapian 2.x installation: XAPIAN_DIR first, then the usual
    // prefixes. /usr needs no special treatment (default compiler paths).
    let prefix = std::env::var_os("XAPIAN_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            ["/usr/local", "/opt/homebrew", "/usr"]
                .iter()
                .map(Path::new)
                .find(|dir| dir.join("include").join("xapian.h").is_file())
                .map(|dir| dir.to_path_buf())
        })
        .unwrap_or_else(|| {
            println!("cargo:warning=xapian.h not found; falling back to default include paths");
            PathBuf::from("/usr")
        });

    let include = prefix.join("include");
    let mut build = cc::Build::new();
    build
        .file("cpp/shim.cpp")
        .cpp(true)
        .include(&include)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-w")
        .warnings(false)
        .opt_level(2);
    build.compile("xapian2_shim");

    println!("cargo:rustc-link-lib=xapian");
    let lib_dir = prefix.join("lib");
    if lib_dir.is_dir() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }

    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-env-changed=XAPIAN_DIR");
}
