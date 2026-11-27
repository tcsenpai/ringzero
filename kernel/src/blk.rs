// virtio-blk on PCI (rcore virtio-drivers): legacy port-I/O config space (0xCF8/0xCFC, bus 0 only), BARs
// identity-mapped uncached (mem::map_mmio), DMA from the identity-mapped engine arena (phys == virt) with
// a bounce ring for every buffer the driver shares (stack, statics and heap are not identity-mapped).
// Polling, one request in flight. Disk wraps it as a byte stream for fatfs (Read/Write/Seek) with a
// one-sector cache, write-through.
use crate::klog;
use core::ptr::NonNull;
use fatfs::{IoBase, IoError, Read, Seek, SeekFrom, Write};
use virtio_drivers::device::blk::VirtIOBlk;
use virtio_drivers::transport::pci::bus::{Command, ConfigurationAccess, DeviceFunction, PciRoot};
use virtio_drivers::transport::pci::{virtio_device_type, PciTransport};
use virtio_drivers::transport::DeviceType;
use virtio_drivers::{BufferDirection, Hal, PhysAddr, PAGE_SIZE};

const SECTOR: usize = 512;

// ---- PCI config space through the legacy I/O ports (works on q35 too) ----

pub struct PortCam;

fn cfg_addr(d: DeviceFunction, off: u8) -> u32 {
    0x8000_0000 | (d.bus as u32) << 16 | (d.device as u32) << 11 | (d.function as u32) << 8 | (off as u32 & 0xfc)
}

impl ConfigurationAccess for PortCam {
    fn read_word(&self, d: DeviceFunction, off: u8) -> u32 {
        let v: u32;
        unsafe {
            core::arch::asm!("out dx, eax", in("dx") 0xcf8u16, in("eax") cfg_addr(d, off), options(nomem, nostack));
            core::arch::asm!("in eax, dx", in("dx") 0xcfcu16, out("eax") v, options(nomem, nostack));
        }
        v
    }
    fn write_word(&mut self, d: DeviceFunction, off: u8, data: u32) {
        unsafe {
            core::arch::asm!("out dx, eax", in("dx") 0xcf8u16, in("eax") cfg_addr(d, off), options(nomem, nostack));
            core::arch::asm!("out dx, eax", in("dx") 0xcfcu16, in("eax") data, options(nomem, nostack));
        }
    }
    unsafe fn unsafe_clone(&self) -> Self {
        PortCam
    }
}

// ---- HAL: DMA and MMIO on the identity-mapped arena ----

pub struct KHal;

const BOUNCE: usize = 256 << 10; // > the largest request (64 KiB) + header + status
static mut BOUNCE_BASE: usize = 0;
static mut BOUNCE_OFF: usize = 0;

fn arena_pages(pages: usize) -> usize {
    let p = crate::mem::sys_mmap_anon(((pages + 1) * PAGE_SIZE) as u64) as usize;
    assert!((p as i64) > 0, "virtio: arena full");
    (p + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

unsafe impl Hal for KHal {
    fn dma_alloc(pages: usize, _dir: BufferDirection) -> (PhysAddr, NonNull<u8>) {
        let p = arena_pages(pages); // zeroed, identity-mapped, never freed (queues are allocated once)
        (p as PhysAddr, NonNull::new(p as *mut u8).unwrap())
    }
    unsafe fn dma_dealloc(_p: PhysAddr, _v: NonNull<u8>, _pages: usize) -> i32 {
        0
    }
    unsafe fn mmio_phys_to_virt(paddr: PhysAddr, size: usize) -> NonNull<u8> {
        crate::mem::map_mmio(paddr as u64, size as u64);
        NonNull::new(paddr as *mut u8).unwrap()
    }
    unsafe fn share(buf: NonNull<[u8]>, dir: BufferDirection) -> PhysAddr {
        // ring allocation: with one request in flight, a 256 KiB ring never reuses a live slot
        let len = buf.len();
        let n = (len + 63) & !63;
        assert!(n <= BOUNCE / 2, "virtio: buffer larger than the bounce ring");
        let (base, off) = unsafe { (BOUNCE_BASE, BOUNCE_OFF) };
        let off = if off + n > BOUNCE { 0 } else { off };
        unsafe { BOUNCE_OFF = off + n };
        let p = base + off;
        if dir != BufferDirection::DeviceToDriver {
            unsafe { core::ptr::copy_nonoverlapping(buf.as_ptr() as *const u8, p as *mut u8, len) };
        }
        p as PhysAddr
    }
    unsafe fn unshare(paddr: PhysAddr, buf: NonNull<[u8]>, dir: BufferDirection) {
        if dir != BufferDirection::DriverToDevice {
            unsafe { core::ptr::copy_nonoverlapping(paddr as *const u8, buf.as_ptr() as *mut u8, buf.len()) };
        }
    }
}

// ---- probe ----

pub type Blk = VirtIOBlk<KHal, PciTransport>;
/// The probed disk for /proc: (PCI device, function, MiB); device 0xff = none.
static mut LOC: (u8, u8, u64) = (0xff, 0, 0);

pub fn location() -> Option<(u8, u8, u64)> {
    let l = unsafe { LOC };
    (l.0 != 0xff).then_some(l)
}

/// First virtio-blk on PCI bus 0 (QEMU q35 puts -device virtio-blk-pci there).
pub fn probe() -> Option<Blk> {
    unsafe { BOUNCE_BASE = arena_pages(BOUNCE / PAGE_SIZE) };
    let mut root = PciRoot::new(PortCam);
    let found = root.enumerate_bus(0).find(|(_, info)| virtio_device_type(info) == Some(DeviceType::Block));
    let (df, info) = found?;
    root.set_command(df, Command::IO_SPACE | Command::MEMORY_SPACE | Command::BUS_MASTER);
    let t = match PciTransport::new::<KHal, _>(&mut root, df) {
        Ok(t) => t,
        Err(e) => {
            klog!(b'E', "disk: virtio-blk at 00:{:02x}.{} (id {:04x}): transport error {:?}", df.device, df.function, info.device_id, e);
            return None;
        }
    };
    match VirtIOBlk::<KHal, _>::new(t) {
        Ok(b) => {
            unsafe { LOC = (df.device, df.function, b.capacity() * SECTOR as u64 >> 20) };
            klog!(b'I', "disk: virtio-blk at PCI 00:{:02x}.{} (device id {:04x}), {} sectors ({} MiB){}", df.device, df.function,
                info.device_id, b.capacity(), b.capacity() * SECTOR as u64 >> 20, if b.readonly() { ", read-only" } else { "" });
            Some(b)
        }
        Err(e) => {
            klog!(b'E', "disk: virtio-blk init failed: {:?}", e);
            None
        }
    }
}

// ---- byte stream for fatfs ----

#[derive(Debug)]
pub enum DiskErr {
    Io,
    Eof,
    WriteZero,
}

impl IoError for DiskErr {
    fn is_interrupted(&self) -> bool {
        false
    }
    fn new_unexpected_eof_error() -> Self {
        DiskErr::Eof
    }
    fn new_write_zero_error() -> Self {
        DiskErr::WriteZero
    }
}

struct Dev {
    blk: Blk,
    pos: u64,
    size: u64,
    lba: u64, // sector in `buf`, u64::MAX = none
    buf: [u8; SECTOR],
}

/// Handle on the one disk (the device lives in DEV), so fs.rs can unmount and mount it again (`!sync`).
pub struct Disk;
static mut DEV: Option<Dev> = None;

const RUN: usize = 128; // sectors per direct transfer (64 KiB)

impl Disk {
    pub fn new(blk: Blk) -> Self {
        let size = blk.capacity() * SECTOR as u64;
        unsafe { DEV = Some(Dev { blk, pos: 0, size, lba: u64::MAX, buf: [0; SECTOR] }) };
        Disk
    }
}

fn dev() -> &'static mut Dev {
    unsafe { (*core::ptr::addr_of_mut!(DEV)).as_mut().expect("disk") }
}

impl Dev {
    fn load(&mut self, lba: u64) -> Result<(), DiskErr> {
        if self.lba != lba {
            self.lba = u64::MAX;
            self.blk.read_blocks(lba as usize, &mut self.buf).map_err(|_| DiskErr::Io)?;
            self.lba = lba;
        }
        Ok(())
    }
}

/// T55: disk size in bytes (/sys/class/block/vda/size), None without a disk.
pub fn size() -> Option<u64> {
    unsafe { (*core::ptr::addr_of!(DEV)).as_ref() }.map(|d| d.size)
}

/// T55: /dev/vda for programs: read the raw disk at `off`; fatfs's position is kept.
pub fn read_raw(off: u64, out: &mut [u8]) -> Result<usize, DiskErr> {
    let d = unsafe { (*core::ptr::addr_of_mut!(DEV)).as_mut() }.ok_or(DiskErr::Io)?;
    let keep = core::mem::replace(&mut d.pos, off);
    let mut n = 0;
    let r = loop {
        match d.read(&mut out[n..]) {
            Ok(0) => break Ok(n),
            Ok(k) => {
                n += k;
                if n == out.len() {
                    break Ok(n);
                }
            }
            Err(e) => break Err(e),
        }
    };
    d.pos = keep;
    r
}

impl IoBase for Disk {
    type Error = DiskErr;
}

impl Dev {
    fn read(&mut self, out: &mut [u8]) -> Result<usize, DiskErr> {
        let left = self.size.saturating_sub(self.pos) as usize;
        let want = out.len().min(left);
        if want == 0 {
            return Ok(0);
        }
        let (lba, off) = (self.pos / SECTOR as u64, self.pos as usize % SECTOR);
        let n = if off == 0 && want >= SECTOR {
            let k = (want / SECTOR).min(RUN) * SECTOR;
            self.blk.read_blocks(lba as usize, &mut out[..k]).map_err(|_| DiskErr::Io)?;
            k
        } else {
            self.load(lba)?;
            let k = want.min(SECTOR - off);
            out[..k].copy_from_slice(&self.buf[off..off + k]);
            k
        };
        self.pos += n as u64;
        Ok(n)
    }
}

impl Dev {
    fn write(&mut self, data: &[u8]) -> Result<usize, DiskErr> {
        let left = self.size.saturating_sub(self.pos) as usize;
        let want = data.len().min(left);
        if want == 0 {
            return Ok(0);
        }
        let (lba, off) = (self.pos / SECTOR as u64, self.pos as usize % SECTOR);
        let n = if off == 0 && want >= SECTOR {
            let k = (want / SECTOR).min(RUN) * SECTOR;
            if self.lba >= lba && self.lba < lba + (k / SECTOR) as u64 {
                self.lba = u64::MAX;
            }
            self.blk.write_blocks(lba as usize, &data[..k]).map_err(|_| DiskErr::Io)?;
            k
        } else {
            self.load(lba)?; // read-modify-write of a partial sector
            let k = want.min(SECTOR - off);
            self.buf[off..off + k].copy_from_slice(&data[..k]);
            let r = self.blk.write_blocks(lba as usize, &self.buf);
            if r.is_err() {
                self.lba = u64::MAX;
                return Err(DiskErr::Io);
            }
            k
        };
        self.pos += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> Result<(), DiskErr> {
        self.blk.flush().map_err(|_| DiskErr::Io) // write-through: only the device cache is left
    }
}

impl Dev {
    fn seek(&mut self, to: SeekFrom) -> Result<u64, DiskErr> {
        let p = match to {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::End(d) => self.size as i64 + d,
            SeekFrom::Current(d) => self.pos as i64 + d,
        };
        if p < 0 || p as u64 > self.size {
            return Err(DiskErr::Io);
        }
        self.pos = p as u64;
        Ok(self.pos)
    }
}

impl Read for Disk {
    fn read(&mut self, out: &mut [u8]) -> Result<usize, DiskErr> {
        dev().read(out)
    }
}

impl Write for Disk {
    fn write(&mut self, data: &[u8]) -> Result<usize, DiskErr> {
        dev().write(data)
    }
    fn flush(&mut self) -> Result<(), DiskErr> {
        dev().flush()
    }
}

impl Seek for Disk {
    fn seek(&mut self, to: SeekFrom) -> Result<u64, DiskErr> {
        dev().seek(to)
    }
}
