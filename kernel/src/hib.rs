// Hibernation (M4.7b): `/hibernate` and every `!poweroff` write the conversation, its snapshot tree and the
// agent's transcript and settings to /var/lib/ringzero/hib/image.bin (snap::hib_save); the next boot resumes it when the
// key matches (the boot snapshot key: model, prefix ids, engine build, KV geometry), with the same state
// hash, so the next answer at temperature 0 is the one the machine would have given before the poweroff.
// The image stays until the next hibernation replaces it or `/hibernate discard` deletes it (and turns the
// poweroff hook off for this session).
use crate::log::SBuf;
use crate::{agent, fs, gram, klog, llm, println, smp, snap};
use alloc::vec::Vec;

use crate::layout::HIB as PATH;
const XVER: u8 = 3;
static mut READY: bool = false;
/// `/hibernate discard`: the next poweroff does not write an image either (a plain `/hibernate` clears it).
static mut SKIP: bool = false;

fn key() -> u64 {
    crate::mstore::fnv(agent::boot_key(), b"ringzero hibernate v1")
}

/// Agent settings and transcript: version, temperature bits, think, constrain, compaction percent,
/// compactions so far, color (v2, T46), tool rounds (v3, T54), transcript.
fn extra() -> Vec<u8> {
    let mut e = alloc::vec![XVER];
    unsafe {
        e.extend_from_slice(&llm::TEMP_BITS.to_le_bytes());
        e.extend_from_slice(&[llm::THINK as u8, gram::ON as u8, agent::COMPACT_PCT as u8]);
        e.extend_from_slice(&agent::N_COMPACT.to_le_bytes());
    }
    e.push(crate::console::COLOR.load(core::sync::atomic::Ordering::Relaxed) as u8);
    e.push(unsafe { agent::ROUNDS } as u8);
    e.extend_from_slice(agent::hist_text().as_bytes());
    e
}

/// Also reads v1 (no color byte: color stays on) and v2 images (no rounds byte: the build default stays).
fn apply(e: &[u8]) {
    let t = match e.first() {
        Some(1) => 16,
        Some(2) => 17,
        Some(&XVER) => 18,
        _ => return,
    };
    if e.len() < t {
        return;
    }
    unsafe {
        llm::TEMP_BITS = u32::from_le_bytes(e[1..5].try_into().unwrap());
        llm::THINK = e[5] as u64;
        gram::ON = e[6] != 0;
        agent::COMPACT_PCT = (e[7] as u64).clamp(1, 95);
        agent::N_COMPACT = u64::from_le_bytes(e[8..16].try_into().unwrap());
    }
    if t >= 17 {
        crate::console::COLOR.store(e[16] != 0, core::sync::atomic::Ordering::Relaxed);
    }
    if t >= 18 {
        unsafe { agent::ROUNDS = (e[17] as usize).clamp(1, agent::ROUNDS_MAX) };
    }
    agent::hist_restore(core::str::from_utf8(&e[t..]).unwrap_or(""));
}

/// Write the image. `why`: "hibernate" or "poweroff".
pub fn save(why: &str) -> bool {
    if !unsafe { READY } || !fs::mounted() || crate::layout::pending().is_some() {
        println!("[hib] no disk, not booted yet or the layout migration is pending: nothing saved");
        return false;
    }
    let t0 = smp::rdtsc();
    let r = snap::hib_save(PATH, key(), &extra());
    let us = smp::tsc_to_us(smp::rdtsc() - t0);
    let mut where_ = SBuf::<96>::new();
    snap::summary(&mut where_);
    match r {
        Ok((bytes, h)) => {
            klog!(b'I', "hib: {} saved {} to {}, {} MiB in {} ms, state hash {:#x}", why, where_.as_str(), PATH, bytes >> 20, us / 1000, h);
            true
        }
        Err(e) => {
            let _ = fs::rm(PATH, false); // no half image
            klog!(b'W', "hib: {} not saved ({}), the next boot starts fresh", why, e);
            false
        }
    }
}

/// Boot, after the root point exists: resume the image if there is a valid one.
pub fn boot() {
    unsafe { READY = true };
    if !fs::mounted() {
        return;
    }
    let t0 = smp::rdtsc();
    match snap::hib_load(PATH, key()) {
        Err("no image") => crate::layout::tmp_clean(),
        Err(e) => {
            klog!(b'W', "hib: image {} not resumed ({}), fresh conversation; /hibernate discard deletes it", PATH, e);
            crate::layout::tmp_clean();
        }
        Ok((stored, e)) => {
            apply(&e);
            let us = smp::tsc_to_us(smp::rdtsc() - t0);
            let h = snap::state_hash();
            let mut w = SBuf::<96>::new();
            snap::summary(&mut w);
            klog!(b'I', "hib: resumed {} in {} ms, state hash {:#x} ({}); /hibernate discard to boot fresh next time",
                w.as_str(), us / 1000, h, if h == stored { "same as saved" } else { "DIFFERS from the saved one" });
            agent::note("[hibernation] resumed after a poweroff");
        }
    }
}

/// T55: `echo disk > /sys/power/state` (sysfs.rs): hibernate and power off, as /hibernate then !poweroff.
pub fn power_disk() -> ! {
    unsafe { SKIP = false };
    klog!(b'I', "hib: disk written to /sys/power/state: hibernate and power off");
    crate::sys::poweroff()
}

/// sys::poweroff, before the disk sync.
pub fn on_poweroff() {
    if unsafe { SKIP } {
        klog!(b'I', "hib: poweroff without an image (/hibernate discard)");
    } else if unsafe { READY } {
        save("poweroff");
    }
}

/// `/hibernate [discard|status]`
pub fn cmd(arg: &str) {
    match arg {
        "" => {
            unsafe { SKIP = false };
            save("hibernate");
        }
        "discard" => {
            unsafe { SKIP = true };
            match fs::rm(PATH, false) {
                Ok(()) => println!("[hib] {} deleted; !poweroff will not write a new one: the next boot starts fresh (/hibernate saves again)", PATH),
                Err(e) => println!("[hib] no image to delete ({}); !poweroff will not write one (/hibernate saves again)", e),
            }
        }
        "status" => match fs::meta(PATH) {
            Ok((size, _)) => println!("[hib] image {} ({} MiB); the next boot resumes it if model, prompt and engine still match", PATH, size >> 20),
            Err(_) => println!("[hib] no image: the next boot starts fresh"),
        },
        _ => println!("usage: /hibernate [discard|status]"),
    }
}
