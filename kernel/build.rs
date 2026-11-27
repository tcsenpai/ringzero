// Assembles the vendored NASM engine and links the objects straight into the kernel.
use std::{env, path::PathBuf, process::Command};

const ASM: &[&str] = &["math", "matmul", "tokenizer", "transformer", "prefill", "cpu", "smp"];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let engine = manifest.join("../engine");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    // KV cache in FP16 by default; RINGZERO_KV=f32 builds the upstream FP32 cache for comparison
    let kv_f32 = env::var("RINGZERO_KV").is_ok_and(|v| v == "f32");
    println!("cargo:rerun-if-env-changed=RINGZERO_KV");
    println!("cargo:rerun-if-env-changed=RINGZERO_EDITION"); // M7: main.rs banner (make MODEL=9b)

    for name in ASM {
        let src = engine.join(format!("{name}.asm"));
        let obj = out.join(format!("{name}.o"));
        let st = Command::new("nasm")
            .args(["-f", "elf64", "-DBARE_METAL", "-DFAST_Q8=1", "-DQWEN35=1"])
            .args(if kv_f32 { &[][..] } else { &["-DKV_F16"][..] })
            .arg(format!("-I{}/", engine.display()))
            .arg("-o").arg(&obj).arg(&src)
            .status()
            .expect("nasm not found");
        assert!(st.success(), "nasm failed on {}", src.display());
        println!("cargo:rustc-link-arg={}", obj.display());
        println!("cargo:rerun-if-changed={}", src.display());
    }
    println!("cargo:rerun-if-changed={}", engine.join("constants.inc").display());
    // engine build id for the boot snapshot cache key (agent.rs): FNV-1a of the sources and the KV flag
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for f in ASM.iter().map(|n| format!("{n}.asm")).chain(["constants.inc".into()]) {
        for b in std::fs::read(engine.join(&f)).expect("engine source").into_iter().chain([kv_f32 as u8]) {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    println!("cargo:rustc-env=RINGZERO_ENGINE_HASH={h:#x}");
    // T38: the manual's vectors (tools/manvec.py, MANUAL_VECTORS=1), linked by agentd.rs; empty = T22 indexing
    println!("cargo:rerun-if-env-changed=RINGZERO_MANVEC");
    println!("cargo:rerun-if-env-changed=RINGZERO_CONF_ASK"); // T28: agent.rs CONF_LO (make iso CONF_ASK=N)
    println!("cargo:rerun-if-env-changed=RINGZERO_ROUNDS"); // T54: agent.rs ROUNDS (make iso ROUNDS=N)
    println!("cargo:rerun-if-env-changed=RINGZERO_LAYOUT_STOP"); // T55: layout.rs crash test builds
    // T52: every other RINGZERO_* option_env! of the kernel (mem.rs heap, wasm.rs memory slot)
    println!("cargo:rerun-if-env-changed=RINGZERO_HEAP_MIB");
    println!("cargo:rerun-if-env-changed=RINGZERO_WASM_MIB");
    let mv = env::var("RINGZERO_MANVEC").ok().filter(|p| !p.is_empty());
    if let Some(p) = &mv {
        println!("cargo:rerun-if-changed={p}");
    }
    std::fs::write(out.join("manvec.bin"), mv.map(|p| std::fs::read(&p).expect("RINGZERO_MANVEC file")).unwrap_or_default()).unwrap();
    println!("cargo:rustc-link-arg=-T{}", manifest.join("linker.ld").display());
    println!("cargo:rerun-if-changed={}", manifest.join("linker.ld").display());
    upy(&manifest, &out);
}

// ---- M5: MicroPython (ports/embed package from `make upy`) + lib/libm_dbl + kernel/upy glue, freestanding ----
fn upy(manifest: &std::path::Path, out: &std::path::Path) {
    let top = manifest.join("../build/micropython");
    let pkg = manifest.join("../build/upy/micropython_embed");
    let glue = manifest.join("upy");
    assert!(pkg.join("genhdr").exists(), "MicroPython embed package missing: run `make upy` first");
    let ls = |d: PathBuf, skip: &[&str]| -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "c") && !skip.iter().any(|s| p.ends_with(s))).collect();
        v.sort();
        v
    };
    let mut srcs = ls(pkg.join("py"), &[]);
    srcs.extend(ls(pkg.join("shared/runtime"), &[]));
    srcs.extend(ls(top.join("lib/libm_dbl"), &["thumb_vfp_sqrt.c"]));
    srcs.extend(ls(glue.clone(), &[])); // glue.c, libc.c (they replace port/embed_util.c and port/mphalport.c)
    let cc = env::var("CC").unwrap_or("cc".into());
    let gcc_inc = String::from_utf8(Command::new(&cc).arg("-print-file-name=include").output().expect("cc").stdout).unwrap();
    // -mcmodel=kernel: the image lives at 0xffffffff80000000; no red zone (interrupts share the stack);
    // SSE2 stays on for doubles (the kernel enables SSE/AVX at boot, Rust itself is soft-float: no floats cross the FFI)
    let flags = ["-std=gnu11", "-O2", "-ffreestanding", "-nostdinc", "-mcmodel=kernel", "-mno-red-zone", "-fno-pic", "-fno-pie",
        "-fno-stack-protector", "-fno-common", "-fno-asynchronous-unwind-tables", "-fno-math-errno", "-DNDEBUG", "-D_LIBC_LIMITS_H_", "-w"]; // gcc's own limits.h, no libc one behind it
    let incs = [PathBuf::from(gcc_inc.trim()), glue.join("include"), glue.clone(), pkg.clone(), pkg.join("port")];
    let (mut objs, mut bad) = (Vec::new(), Vec::new());
    for chunk in srcs.chunks(16) {
        let kids: Vec<_> = chunk.iter().map(|s| {
            let o = out.join(format!("upy_{}_{}.o", s.parent().unwrap().file_name().unwrap().to_str().unwrap(),
                s.file_stem().unwrap().to_str().unwrap()));
            let mut c = Command::new(&cc);
            c.args(flags);
            incs.iter().for_each(|i| { c.arg("-isystem").arg(i); });
            c.arg("-c").arg(s).arg("-o").arg(&o);
            objs.push(o);
            (s.clone(), c.spawn().expect("cc"))
        }).collect();
        for (s, mut k) in kids {
            if !k.wait().unwrap().success() {
                bad.push(s.display().to_string());
            }
        }
    }
    assert!(bad.is_empty(), "cc failed on {}", bad.join(" "));
    let lib = out.join("libupy.a");
    let _ = std::fs::remove_file(&lib);
    assert!(Command::new("ar").arg("crs").arg(&lib).args(&objs).status().expect("ar").success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=upy");
    println!("cargo:rerun-if-changed={}", glue.display());
    println!("cargo:rerun-if-changed={}", pkg.join("genhdr").display());
    println!("cargo:rerun-if-env-changed=RINGZERO_UPY_HEAP_MIB");
}
