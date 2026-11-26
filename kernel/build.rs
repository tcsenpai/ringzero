// Assembles the vendored NASM engine and links the objects straight into the kernel.
use std::{env, path::PathBuf, process::Command};

const ASM: &[&str] = &["math", "matmul", "tokenizer", "transformer", "cpu", "smp"];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let engine = manifest.join("../engine");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    for name in ASM {
        let src = engine.join(format!("{name}.asm"));
        let obj = out.join(format!("{name}.o"));
        let st = Command::new("nasm")
            .args(["-f", "elf64", "-DBARE_METAL", "-DFAST_Q8=1", "-DQWEN35=1"])
            .arg(format!("-I{}/", engine.display()))
            .arg("-o").arg(&obj).arg(&src)
            .status()
            .expect("nasm not found");
        assert!(st.success(), "nasm failed on {}", src.display());
        println!("cargo:rustc-link-arg={}", obj.display());
        println!("cargo:rerun-if-changed={}", src.display());
    }
    println!("cargo:rerun-if-changed={}", engine.join("constants.inc").display());
    println!("cargo:rustc-link-arg=-T{}", manifest.join("linker.ld").display());
    println!("cargo:rerun-if-changed={}", manifest.join("linker.ld").display());
}
