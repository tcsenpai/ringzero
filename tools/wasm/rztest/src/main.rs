// rztest: checks of ringzero's WASI preview1 runtime (M5b), built for wasm32-wasip1.
//   rztest check A "B C"   every check, ends with "rztest: PASS n/n" (exit 1 on a failure)
//   rztest stdin           count the bytes and lines of stdin (`!wasm rztest stdin < /file`)
//   rztest bench           CPU bench (sieve + integer hash), prints ms; the same source runs natively for the ratio
//   rztest loop            runaway loop (time budget, Ctrl-C)
//   rztest mem MIB         allocate and touch MIB MiB (memory cap)
//   rztest sleep MS        std::thread::sleep (poll_oneoff)
//   rztest escape          try to leave the preopened dir
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_os = "wasi")]
#[link(wasm_import_module = "wasi_snapshot_preview1")]
unsafe extern "C" {
    fn random_get(buf: *mut u8, len: usize) -> i32;
    fn sock_accept(fd: i32, flags: i32, out: *mut i32) -> i32; // not implemented by ringzero: ENOSYS (52)
}

fn nosys() -> bool {
    #[cfg(target_os = "wasi")]
    return (0..2).all(|_| unsafe { sock_accept(3, 0, &mut 0) } == 52);
    #[cfg(not(target_os = "wasi"))]
    true
}

fn random(buf: &mut [u8]) -> bool {
    #[cfg(target_os = "wasi")]
    return unsafe { random_get(buf.as_mut_ptr(), buf.len()) } == 0;
    #[cfg(not(target_os = "wasi"))]
    {
        let _ = buf;
        true
    }
}

fn bench() -> u64 {
    let t = Instant::now();
    let n = 2_000_000;
    let mut sieve = vec![true; n + 1];
    let mut count = 0u64;
    for i in 2..=n {
        if sieve[i] {
            count += 1;
            let mut j = if i <= n / i { i * i } else { n + 1 }; // i * i overflows a 32-bit usize on wasm32
            while j <= n {
                sieve[j] = false;
                j += i;
            }
        }
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for i in 0..20_000_000u64 {
        h = (h ^ i).wrapping_mul(0x100_0000_01b3).rotate_left(5);
    }
    let ms = t.elapsed().as_millis() as u64;
    println!("bench: {} primes <= {}, hash {:016x}, {} ms", count, n, h, ms);
    ms
}

fn escape() -> bool {
    let mut refused = true;
    for p in ["../x", "/../x", "a/../../x", "../../etc/passwd"] {
        let r = fs::File::open(p);
        println!("open {:?}: {}", p, match &r { Ok(_) => "OPENED (escape!)".into(), Err(e) => format!("refused ({})", e) });
        refused &= r.is_err();
    }
    let w = fs::write("../escape.txt", b"x");
    println!("write \"../escape.txt\": {}", match &w { Ok(_) => "WRITTEN (escape!)".into(), Err(e) => format!("refused ({})", e) });
    refused && w.is_err()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("stdin") => {
            let mut b = Vec::new();
            std::io::stdin().read_to_end(&mut b).unwrap();
            println!("stdin: {} bytes, {} lines", b.len(), b.iter().filter(|&&c| c == b'\n').count());
            return;
        }
        Some("bench") => {
            bench();
            return;
        }
        Some("loop") => {
            println!("looping forever");
            let mut x = 0u64;
            loop {
                x = std::hint::black_box(x.wrapping_add(1));
            }
        }
        Some("mem") => {
            let mib: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
            let mut v: Vec<Vec<u8>> = Vec::new();
            for i in 0..mib {
                let mut c = Vec::new();
                if c.try_reserve_exact(1 << 20).is_err() {
                    println!("mem: allocation failed after {} MiB", i);
                    std::process::exit(3);
                }
                c.resize(1 << 20, i as u8);
                v.push(c);
            }
            println!("mem: {} MiB allocated and touched", v.len());
            return;
        }
        Some("sleep") => {
            let ms: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            let t = Instant::now();
            std::thread::sleep(std::time::Duration::from_millis(ms));
            println!("slept {} ms (asked {})", t.elapsed().as_millis(), ms);
            return;
        }
        Some("escape") => {
            std::process::exit(if escape() { 0 } else { 1 });
        }
        Some("check") => {}
        _ => {
            println!("usage: rztest check|stdin|bench|loop|mem MIB|sleep MS|escape");
            return;
        }
    }

    let (mut ok, mut n) = (0, 0);
    let mut t = |name: &str, pass: bool, detail: String| {
        n += 1;
        ok += pass as u32;
        println!("[{}] {}: {}", if pass { "ok" } else { "FAIL" }, name, detail);
    };
    println!("hello from wasm32-wasip1 (rztest {})", env!("CARGO_PKG_VERSION"));
    eprintln!("this line goes to stderr");
    t("args", args.len() == 4 && args[2] == "A" && args[3] == "B C", format!("{:?}", args));
    t("env", std::env::var("HOME").as_deref() == Ok("/"), format!("HOME={:?}", std::env::var("HOME")));

    let d = "/rztest";
    let _ = fs::remove_dir_all(d);
    t("mkdir", fs::create_dir(d).is_ok(), d.into());
    let f = format!("{}/a.txt", d);
    let body = "line one\nline two\n";
    t("write", fs::write(&f, body).is_ok(), format!("{} ({} B)", f, body.len()));
    let back = fs::read_to_string(&f).unwrap_or_default();
    t("read back", back == body, format!("{:?}", back));
    let app = fs::OpenOptions::new().append(true).open(&f).and_then(|mut h| h.write_all(b"line three\n"));
    let size = fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
    t("append + metadata", app.is_ok() && size == 29, format!("size {}", size));
    let mut word = [0u8; 3];
    let seek = fs::File::open(&f).and_then(|mut h| {
        h.seek(SeekFrom::Start(5))?;
        h.read_exact(&mut word)
    });
    t("seek + read", seek.is_ok() && &word == b"one", format!("{:?}", String::from_utf8_lossy(&word)));
    let rel = fs::read_to_string("rztest/a.txt").map(|s| s.len()).unwrap_or(0);
    t("relative path (cwd /)", rel == 29, format!("{} B", rel));
    let _ = fs::write(format!("{}/b.bin", d), [0u8, 1, 2, 255]);
    let _ = fs::create_dir(format!("{}/sub", d));
    let mut names: Vec<String> = fs::read_dir(d).map(|r| r.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect()).unwrap_or_default();
    names.sort();
    t("read_dir", names == ["a.txt", "b.bin", "sub"], format!("{:?}", names));
    let ty = fs::metadata(format!("{}/sub", d)).map(|m| m.is_dir()).unwrap_or(false);
    t("metadata is_dir", ty, format!("{}/sub", d));
    let mv = fs::rename(format!("{}/b.bin", d), format!("{}/sub/c.bin", d));
    let moved = fs::read(format!("{}/sub/c.bin", d)).unwrap_or_default();
    t("rename", mv.is_ok() && moved == [0, 1, 2, 255] && fs::metadata(format!("{}/b.bin", d)).is_err(), format!("{:?}", mv));
    t("remove_file", fs::remove_file(format!("{}/sub/c.bin", d)).is_ok(), "sub/c.bin".into());
    t("remove_dir", fs::remove_dir(format!("{}/sub", d)).is_ok() && fs::metadata(format!("{}/sub", d)).is_err(), "sub".into());
    t("missing file", fs::read("/no/such/file").is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound), "NotFound".into());
    // T55: ringzero's data is EACCES (PermissionDenied), the virtual trees EROFS (ReadOnlyFilesystem)
    let lib = fs::write("/var/lib/ringzero/rztest.txt", b"x");
    t("/var/lib/ringzero is read-only", lib.as_ref().is_err_and(|e| e.kind() == std::io::ErrorKind::PermissionDenied), format!("{:?}", lib.err().map(|e| e.kind())));
    let sys = fs::write("/sys/rztest.txt", b"x");
    t("/sys is a read-only file system", sys.as_ref().is_err_and(|e| e.raw_os_error() == Some(69)), format!("{:?}", sys.err()));
    let vda = fs::OpenOptions::new().write(true).open("/dev/vda");
    t("/dev/vda is read-only", vda.as_ref().is_err_and(|e| e.raw_os_error() == Some(69)), format!("{:?}", vda.err()));
    let zero = fs::File::open("/dev/zero").and_then(|mut f| { let mut b = [1u8; 64]; f.read_exact(&mut b).map(|_| b) });
    t("/dev/zero reads zeros", zero.as_ref().is_ok_and(|b| b.iter().all(|&x| x == 0)), format!("{:?}", zero.err()));
    let null = fs::write("/dev/null", vec![7u8; 1 << 20]);
    t("/dev/null takes 1 MiB", null.is_ok() && fs::read("/dev/null").is_ok_and(|v| v.is_empty()), format!("{:?}", null.err()));
    t("sandbox: '..' above / refused", escape(), "see the open lines above".into());

    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    t("clock realtime", now > 1_700_000_000, format!("{} s since the epoch", now));
    let i0 = Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(50));
    let el = i0.elapsed().as_millis();
    t("clock monotonic + sleep 50 ms", (50..500).contains(&el), format!("{} ms", el));
    let (mut a, mut b) = ([0u8; 16], [0u8; 16]);
    t("random_get", random(&mut a) && random(&mut b) && a != b && a != [0; 16], format!("{:02x?}", &a[..4]));
    t("unimplemented call -> ENOSYS", nosys(), "sock_accept twice (the kernel logs it once)".into());
    let _ = fs::remove_dir_all(d);

    println!("rztest: {} {}/{}", if ok == n { "PASS" } else { "FAIL" }, ok, n);
    std::process::exit(if ok == n { 0 } else { 1 });
}
