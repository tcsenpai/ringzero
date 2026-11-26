# ringzero build. Run on the Linux build box (needs cargo, nasm, xorriso, git, cc, qemu).
LIMINE_TAG ?= v11.2.1-binary
LIMINE_URL ?= https://codeberg.org/Limine/Limine.git
# Model choice (README "Which model for which machine"): MODEL=2b|4b, QUANT=q8|q4.
#   2b q8 (default)  Qwen3.5-2B Q8_0, .asmbin v3, 6G guest
#   2b q4            Qwen3.5-2B Q4_0 mixed (the Q4 kernel test bench, not a product target)
#   4b q4            Qwen3.5-4B Q4_0 mixed (embed/lm_head Q8_0, small tensors f32), .asmbin v4, 6G guest
#   4b q8            Qwen3.5-4B Q8_0, a 4.1 GiB module, 8G guest
#   9b               not built: docs/FEASIBILITY-q4-qwen35-9b.md
# MODEL=path/to/file.asmbin boots any exported file. `make model` fetches + exports only the selected one.
MODEL      ?= 2b
QUANT      ?= q8
EMBED      ?= models/qwen3-embed-0.6b-q8_0-v3.asmbin
ifneq ($(filter 2b 4b,$(MODEL)),)
MODEL_REPO := Qwen/Qwen3.5-$(shell echo $(MODEL) | tr b B)
ifeq ($(QUANT),q4)
MODEL_FILE := models/qwen35-$(MODEL)-q4_0-v4.asmbin
MODEL_EXPORT := --q4 all
EDITION    := ringzero Qwen35@$(MODEL).q4-Qwen3@0.6b (QweQwe)
else ifeq ($(QUANT),q8)
MODEL_FILE := models/qwen35-$(MODEL)-q8_0-v3.asmbin
EDITION    := ringzero Qwen35@$(MODEL)-Qwen3@0.6b (QweQwe)
else
$(error QUANT must be q8 or q4)
endif
ifeq ($(MODEL),4b)
MODEL_HF_FILES := config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
                  model.safetensors-00001-of-00002.safetensors model.safetensors-00002-of-00002.safetensors
ifeq ($(QUANT),q8)
# 4.1 GiB weights: the module needs a contiguous run above 4 GiB (docs/M7_Q4_RESULT.md, memory)
GUEST_MEM  ?= 8G
MIN_AVAIL_MB ?= 11264
endif
endif
else
MODEL_FILE := $(MODEL)
EDITION    ?= ringzero (custom model)
endif
# guest RAM: 6 GiB since the embedder (see docs/EMBEDDER_RESULT.md, memory map); run-* refuse below MIN_AVAIL_MB
GUEST_MEM  ?= 6G
MIN_AVAIL_MB ?= 8192
OVMF_CODE  ?= /usr/share/edk2/ovmf/OVMF_CODE.fd
OVMF_VARS  ?= /usr/share/edk2/ovmf/OVMF_VARS.fd
QEMU_DISPLAY ?= none
# FAT32 data disk (virtio-blk, mounted at / in the guest): `make disk` creates it once and never overwrites it
DISK       ?= build/disk.img
DISK_MB    ?= 512
# run-local's own disk, so it can run next to run-remote (QEMU's image lock refuses two writers on one file)
LOCAL_DISK ?= build/disk-local.img
DRIVE   = -drive file=$(1),if=none,id=disk0,format=raw -device virtio-blk-pci,drive=disk0

BUILD  := build
KERNEL := kernel/target/x86_64-unknown-none/release/ringzero-kernel
ISO    := $(BUILD)/ringzero.iso
QEMU   := qemu-system-x86_64 -enable-kvm -cpu host -M q35 -m $(GUEST_MEM) -smp 8 \
          -cdrom $(ISO) -boot d -serial stdio -display $(QEMU_DISPLAY) -no-reboot $(call DRIVE,$(DISK))

.PHONY: kernel iso model disk run-bios run-uefi run-remote run-local clean
.DELETE_ON_ERROR:

# Qwen3.5-2B -> $(MODEL) and Qwen3-Embedding-0.6B -> $(EMBED) from Hugging Face, each skipped when the .asmbin
# exists (HF_DIR / EMBED_HF_DIR = dir reuses a download, see tools/fetch_model.sh)
EMBED_HF_DIR ?=
model:
	HF_REPO=$(MODEL_REPO) EXPORT_ARGS="$(MODEL_EXPORT)" $(if $(MODEL_HF_FILES),HF_FILES="$(MODEL_HF_FILES)") \
	    tools/fetch_model.sh $(MODEL_FILE)
	HF_DIR="$(EMBED_HF_DIR)" HF_REPO=Qwen/Qwen3-Embedding-0.6B EXPORT_ARGS="--v3 --max-seq-len 1024" \
	    HF_FILES="config.json model.safetensors tokenizer.json tokenizer_config.json" tools/fetch_model.sh $(EMBED)

kernel:
	cd kernel && RINGZERO_EDITION="$(EDITION)" cargo build --release -j 8

$(BUILD)/limine/limine:
	rm -rf $(BUILD)/limine
	git clone --depth=1 --branch $(LIMINE_TAG) $(LIMINE_URL) $(BUILD)/limine
	$(MAKE) -C $(BUILD)/limine

iso: kernel $(BUILD)/limine/limine
	rm -rf $(BUILD)/iso_root
	mkdir -p $(BUILD)/iso_root/boot/limine $(BUILD)/iso_root/EFI/BOOT
	cp $(KERNEL) $(BUILD)/iso_root/boot/kernel
	cp --reflink=auto $(MODEL_FILE) $(BUILD)/iso_root/boot/model.asmbin
	cp --reflink=auto $(EMBED) $(BUILD)/iso_root/boot/embed.asmbin
	cp limine.conf $(addprefix $(BUILD)/limine/,limine-bios.sys limine-bios-cd.bin limine-uefi-cd.bin) \
	   $(BUILD)/iso_root/boot/limine/
	cp $(BUILD)/limine/BOOTX64.EFI $(BUILD)/iso_root/EFI/BOOT/
	xorriso -as mkisofs -iso-level 3 -R -r -J -b boot/limine/limine-bios-cd.bin \
	    -no-emul-boot -boot-load-size 4 -boot-info-table \
	    --efi-boot boot/limine/limine-uefi-cd.bin -efi-boot-part --efi-boot-image \
	    --protective-msdos-label $(BUILD)/iso_root -o $(ISO)
	$(BUILD)/limine/limine bios-install $(ISO)
	rm -rf $(BUILD)/iso_root

# whole-volume FAT32, no partition table (mtools on the host: mdir -i build/disk.img ::/)
disk: $(DISK)
$(BUILD)/%.img:
	mkdir -p $(BUILD)
	@if [ -e $@ ]; then echo "$@ exists, not touching it"; exit 0; fi; \
	 truncate -s $(DISK_MB)M $@.tmp && \
	 { mkfs.vfat -F 32 -n RINGZERO $@.tmp >/dev/null || mformat -F -v RINGZERO -i $@.tmp ::; } && mv -n $@.tmp $@ && \
	 echo "created $@ ($(DISK_MB) MiB FAT32)"

run-bios: iso $(DISK)
	$(QEMU)

run-uefi: iso $(DISK)
	cp $(OVMF_VARS) $(BUILD)/ovmf_vars.fd
	$(QEMU) -drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
	        -drive if=pflash,format=raw,file=$(BUILD)/ovmf_vars.fd

# Headless guest for remote use: serial console on TCP 4555 (nc HOST 4555), framebuffer on VNC :1
# (port 5901, password VNC_PASS; LAN-only throwaway test VM). Everything the guest prints on serial
# (including the echoed input) is appended to build/remote-serial.log; QEMU monitor (screendump) on
# build/remote-monitor.sock. VNC_KEYMAP: keysym -> scancode map for VNC clients without the QEMU
# extended key event (macOS Screen Sharing); match it to the guest layout (`kbd`).
REMOTE_SMP ?= 8
VNC_PASS   ?= aios
# a second (test) instance: make run-remote REMOTE_PORT=4556 REMOTE_VNC=2 REMOTE_NAME=test
REMOTE_PORT ?= 4555
REMOTE_VNC  ?= 1
REMOTE_NAME ?= remote
QEMU_KEYMAPS ?= /usr/share/qemu/keymaps
VNC_KEYMAP ?= $(BUILD)/it-ringzero
$(BUILD)/it-ringzero: tools/keymaps/it-altgr
	mkdir -p $(BUILD) && cat $(QEMU_KEYMAPS)/it $< > $@

run-remote: iso $(DISK) $(filter $(BUILD)/%,$(VNC_KEYMAP))
	@avail=$$(awk '/MemAvailable/ {print int($$2/1024)}' /proc/meminfo); \
	 if [ $$avail -lt $(MIN_AVAIL_MB) ]; then echo "refusing: only $$avail MiB available (want > $(MIN_AVAIL_MB) for a $(GUEST_MEM) guest)"; exit 1; fi
	qemu-system-x86_64 -enable-kvm -cpu host -M q35 -m $(GUEST_MEM) -smp $(REMOTE_SMP) \
	    -cdrom $(ISO) -boot d -no-reboot -vga std \
	    -object secret,id=vncpw,data=$(VNC_PASS) -display vnc=0.0.0.0:$(REMOTE_VNC),password-secret=vncpw -k $(VNC_KEYMAP) \
	    -chardev socket,id=ser0,host=0.0.0.0,port=$(REMOTE_PORT),server=on,wait=off,logfile=$(BUILD)/$(REMOTE_NAME)-serial.log,logappend=on \
	    -serial chardev:ser0 -monitor unix:$(BUILD)/$(REMOTE_NAME)-monitor.sock,server,nowait $(call DRIVE,$(DISK))

# The same VM in a QEMU window on the Fedora desktop (PS/2 keys from the window, guest layout `!kbd it|us`),
# serial on this terminal (Ctrl-C goes to the guest; close the window to quit) and in build/local-serial.log.
# Own disk image by default; LOCAL_DISK=build/disk.img shares run-remote's disk while run-remote is stopped.
LOCAL_DISPLAY ?= gtk
run-local: iso $(LOCAL_DISK)
	@avail=$$(awk '/MemAvailable/ {print int($$2/1024)}' /proc/meminfo); \
	 if [ $$avail -lt $(MIN_AVAIL_MB) ]; then echo "refusing: only $$avail MiB available (want > $(MIN_AVAIL_MB) for a $(GUEST_MEM) guest)"; exit 1; fi
	qemu-system-x86_64 -enable-kvm -cpu host -M q35 -m $(GUEST_MEM) -smp $(REMOTE_SMP) \
	    -cdrom $(ISO) -boot d -no-reboot -vga std -display $(LOCAL_DISPLAY) -name ringzero \
	    -chardev stdio,id=ser0,signal=off,logfile=$(BUILD)/local-serial.log,logappend=on -serial chardev:ser0 \
	    -monitor unix:$(BUILD)/local-monitor.sock,server,nowait $(call DRIVE,$(LOCAL_DISK))

clean:
	rm -rf $(BUILD) kernel/target
