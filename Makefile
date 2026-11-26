# AIOS build. Run on the Linux build box (needs cargo, nasm, xorriso, git, cc, qemu).
LIMINE_TAG ?= v11.2.1-binary
LIMINE_URL ?= https://codeberg.org/Limine/Limine.git
MODEL      ?= $(HOME)/coding/qwenbaremetal/models/qwen35-2b-q8_0-v3.asmbin
OVMF_CODE  ?= /usr/share/edk2/ovmf/OVMF_CODE.fd
OVMF_VARS  ?= /usr/share/edk2/ovmf/OVMF_VARS.fd
QEMU_DISPLAY ?= none

BUILD  := build
KERNEL := kernel/target/x86_64-unknown-none/release/aios-kernel
ISO    := $(BUILD)/aios.iso
QEMU   := qemu-system-x86_64 -enable-kvm -cpu host -M q35 -m 4G -smp 8 \
          -cdrom $(ISO) -boot d -serial stdio -display $(QEMU_DISPLAY) -no-reboot

.PHONY: kernel iso run-bios run-uefi run-remote clean
.DELETE_ON_ERROR:

kernel:
	cd kernel && cargo build --release -j 8

$(BUILD)/limine/limine:
	rm -rf $(BUILD)/limine
	git clone --depth=1 --branch $(LIMINE_TAG) $(LIMINE_URL) $(BUILD)/limine
	$(MAKE) -C $(BUILD)/limine

iso: kernel $(BUILD)/limine/limine
	rm -rf $(BUILD)/iso_root
	mkdir -p $(BUILD)/iso_root/boot/limine $(BUILD)/iso_root/EFI/BOOT
	cp $(KERNEL) $(BUILD)/iso_root/boot/kernel
	cp --reflink=auto $(MODEL) $(BUILD)/iso_root/boot/model.asmbin
	cp limine.conf $(addprefix $(BUILD)/limine/,limine-bios.sys limine-bios-cd.bin limine-uefi-cd.bin) \
	   $(BUILD)/iso_root/boot/limine/
	cp $(BUILD)/limine/BOOTX64.EFI $(BUILD)/iso_root/EFI/BOOT/
	xorriso -as mkisofs -R -r -J -b boot/limine/limine-bios-cd.bin \
	    -no-emul-boot -boot-load-size 4 -boot-info-table \
	    --efi-boot boot/limine/limine-uefi-cd.bin -efi-boot-part --efi-boot-image \
	    --protective-msdos-label $(BUILD)/iso_root -o $(ISO)
	$(BUILD)/limine/limine bios-install $(ISO)
	rm -rf $(BUILD)/iso_root

run-bios: iso
	$(QEMU)

run-uefi: iso
	cp $(OVMF_VARS) $(BUILD)/ovmf_vars.fd
	$(QEMU) -drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
	        -drive if=pflash,format=raw,file=$(BUILD)/ovmf_vars.fd

# Headless guest for remote use: serial console on TCP 4555 (nc HOST 4555), framebuffer on VNC :1
# (port 5901, password VNC_PASS; LAN-only throwaway test VM), QEMU monitor on build/remote.mon.
REMOTE_SMP ?= 8
VNC_PASS   ?= aios
run-remote: iso
	@avail=$$(awk '/MemAvailable/ {print int($$2/1024)}' /proc/meminfo); \
	 if [ $$avail -lt 6144 ]; then echo "refusing: only $$avail MiB available (want > 6144)"; exit 1; fi
	qemu-system-x86_64 -enable-kvm -cpu host -M q35 -m 4G -smp $(REMOTE_SMP) \
	    -cdrom $(ISO) -boot d -no-reboot -vga std \
	    -object secret,id=vncpw,data=$(VNC_PASS) -display vnc=0.0.0.0:1,password-secret=vncpw \
	    -serial tcp:0.0.0.0:4555,server,nowait -monitor unix:$(BUILD)/remote.mon,server,nowait

clean:
	rm -rf $(BUILD) kernel/target
