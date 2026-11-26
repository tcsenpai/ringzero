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

.PHONY: kernel iso run-bios run-uefi clean
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

clean:
	rm -rf $(BUILD) kernel/target
