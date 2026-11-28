# ringzero build. Run on the Linux build box (needs cargo, nasm, xorriso, git, cc, qemu).
LIMINE_TAG ?= v11.2.1-binary
LIMINE_URL ?= https://codeberg.org/Limine/Limine.git
# Model choice (INFO.md "Which model for which machine"): MODEL=2b|4b, QUANT=q8|q4|q4all.
#   q8     every matrix Q8_0 (.asmbin v3)
#   q4     MLP (gate/up/down) Q4_0, attention + DeltaNet projections, embed/lm_head Q8_0 (.asmbin v4)
#   q4all  every projection Q4_0, embed/lm_head Q8_0 (smallest, lowest quality; docs/M7_Q4_RESULT.md)
#   (norms, conv1d, A_log, dt_bias, in_proj_a/b stay f32 in all of them)
#   2b q8 (default)  6G guest | 2b q4/q4all: the Q4 kernel test bench, not a product target
#   4b q4            Qwen3.5-4B, 3.1 GiB module | 4b q4all 2.5 GiB | 4b q8 4.2 GiB module: all 8G guests
#   9b               not built: docs/FEASIBILITY-q4-qwen35-9b.md
# MODEL=path/to/file.asmbin boots any exported file. `make model` fetches + exports only the selected one.
MODEL      ?= 2b
QUANT      ?= q8
EMBED      ?= models/qwen3-embed-0.6b-q8_0-v3.asmbin
ifneq ($(filter 2b 4b,$(MODEL)),)
MODEL_REPO := Qwen/Qwen3.5-$(shell echo $(MODEL) | tr b B)
ifeq ($(QUANT),q4)
MODEL_FILE := models/qwen35-$(MODEL)-q4_0mlp-v4.asmbin
MODEL_EXPORT := --q4 mlp,mlp_down
EDITION    := ringzero Qwen35@$(MODEL).q4-Qwen3@0.6b (QweQwe)
else ifeq ($(QUANT),q4all)
MODEL_FILE := models/qwen35-$(MODEL)-q4_0-v4.asmbin
MODEL_EXPORT := --q4 all
EDITION    := ringzero Qwen35@$(MODEL).q4all-Qwen3@0.6b (QweQwe)
else ifeq ($(QUANT),q8)
MODEL_FILE := models/qwen35-$(MODEL)-q8_0-v3.asmbin
EDITION    := ringzero Qwen35@$(MODEL)-Qwen3@0.6b (QweQwe)
else
$(error QUANT must be q8, q4 or q4all)
endif
ifeq ($(MODEL),4b)
MODEL_HF_FILES := config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
                  model.safetensors-00001-of-00002.safetensors model.safetensors-00002-of-00002.safetensors
ifeq ($(QUANT),q8)
# 4.1 GiB weights: the module needs a contiguous run above 4 GiB (docs/M7_Q4_RESULT.md, memory)
MIN_AVAIL_MB ?= 11264
endif
# every 4B option: 8G. At 6G the 4B q4all panics at boot ("snapshot pool: no room for the prefix") since the
# M5 / M5b arena users; 8G gives the 1024 MiB pool (docs/MODEL_CHOICE.md)
GUEST_MEM  ?= 8G
MIN_AVAIL_MB ?= 10240
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

.PHONY: kernel upy iso model disk wasm-apps beta run-bios run-uefi run-remote run-local clean
.DELETE_ON_ERROR:

# Qwen3.5-2B -> $(MODEL) and Qwen3-Embedding-0.6B -> $(EMBED) from Hugging Face, each skipped when the .asmbin
# exists (HF_DIR / EMBED_HF_DIR = dir reuses a download, see tools/fetch_model.sh)
EMBED_HF_DIR ?=
model:
	HF_REPO=$(MODEL_REPO) EXPORT_ARGS="$(MODEL_EXPORT)" $(if $(MODEL_HF_FILES),HF_FILES="$(MODEL_HF_FILES)") \
	    tools/fetch_model.sh $(MODEL_FILE)
	HF_DIR="$(EMBED_HF_DIR)" HF_REPO=Qwen/Qwen3-Embedding-0.6B EXPORT_ARGS="--v3 --max-seq-len 1024" \
	    HF_FILES="config.json model.safetensors tokenizer.json tokenizer_config.json" tools/fetch_model.sh $(EMBED)

# T38: MANUAL_VECTORS=1 (default) links the manual's vectors into the kernel: pass 1 builds an ISO without them,
# boots it headless once (tools/manvec.py: `/man dump`, the kernel's own Q8 embedder) and writes $(MANVEC);
# pass 2 is the normal kernel build with that file. MANUAL_VECTORS=0: no vectors, the manual is embedded in idle
# time at the first boot (T22). The kernel checks the key at boot and falls back to T22 on any mismatch.
MANUAL_VECTORS ?= 1
MANVEC := $(BUILD)/manvec.bin
ifeq ($(MANUAL_VECTORS),1)
kernel: $(MANVEC)
MANVEC_PATH := $(abspath $(MANVEC))
endif
$(MANVEC): docs/MANUAL.md $(EMBED) $(wildcard engine/*.asm engine/*.inc) kernel/src/embed.rs kernel/src/memidx.rs kernel/src/memrank.rs kernel/src/llm.rs
	$(MAKE) iso MANUAL_VECTORS=0 ISO=$(BUILD)/manvec-pass1.iso
	MEM=$(GUEST_MEM) python3 tools/manvec.py $(BUILD)/manvec-pass1.iso $@
	rm -rf $(BUILD)/manvec-pass1.iso $@.boot

# T28: ask y/N before a call of a tool that changes the machine when its confidence margin is below CONF_ASK percent
# (0..100, clamped; default 80; 25 = the tune-split pick of docs/T28_RESULT.md)
CONF_ASK ?=
# T54: tool rounds per question before the agent answers without tools (1..20, clamped; default 10; !set rounds N)
ROUNDS ?=
# T55 crash test builds only: halt after migration step N of an old disk (layout.rs); empty = never
LAYOUT_STOP ?=
kernel: upy
	cd kernel && RINGZERO_EDITION="$(EDITION)" RINGZERO_MANVEC="$(MANVEC_PATH)" RINGZERO_CONF_ASK="$(CONF_ASK)" RINGZERO_ROUNDS="$(ROUNDS)" RINGZERO_LAYOUT_STOP="$(LAYOUT_STOP)" cargo build --release -j 8

# M5: MicroPython (MIT), fetched at a pinned tag + commit and turned into the ports/embed package that
# kernel/build.rs compiles freestanding (config: kernel/upy/mpconfigport.h). Needs python3 on the build box.
UPY_TAG    ?= v1.29.0
UPY_COMMIT ?= 0fd6c573ea815774668bbb16b8e197c8822368b2
UPY_URL    ?= https://github.com/micropython/micropython.git
UPY_STAMP  := $(BUILD)/upy/micropython_embed/genhdr/qstrdefs.generated.h
upy: $(UPY_STAMP)
$(BUILD)/micropython/py/py.mk:
	rm -rf $(BUILD)/micropython
	git -c advice.detachedHead=false clone -q --depth=1 --branch $(UPY_TAG) $(UPY_URL) $(BUILD)/micropython
	test "$$(git -C $(BUILD)/micropython rev-parse HEAD)" = "$(UPY_COMMIT)" || { echo "MicroPython $(UPY_TAG) is not commit $(UPY_COMMIT)"; exit 1; }
$(UPY_STAMP): $(BUILD)/micropython/py/py.mk kernel/upy/mpconfigport.h kernel/upy/micropython_embed.mk
	$(MAKE) -s -C kernel/upy -f micropython_embed.mk MICROPYTHON_TOP=$(abspath $(BUILD)/micropython) \
	    BUILD=$(abspath $(BUILD)/upy/build-embed) PACKAGE_DIR=$(abspath $(BUILD)/upy/micropython_embed)
	touch $@

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
	    --protective-msdos-label $(if $(SOURCE_DATE_EPOCH),--set_all_file_dates =$(SOURCE_DATE_EPOCH)) $(BUILD)/iso_root -o $(ISO)
	$(BUILD)/limine/limine bios-install $(ISO)
	@# make beta: bios-install writes a random MBR disk id (srand(time(NULL))); a fixed one keeps the ISO reproducible
	$(if $(SOURCE_DATE_EPOCH),printf 'RZ01' | dd of=$(ISO) bs=1 seek=440 conv=notrunc status=none)
	rm -rf $(BUILD)/iso_root

# whole-volume FAT32, no partition table (mtools on the host: mdir -i build/disk.img ::/)
disk: $(DISK)
$(BUILD)/%.img:
	mkdir -p $(BUILD)
	@if [ -e $@ ]; then echo "$@ exists, not touching it"; exit 0; fi; \
	 truncate -s $(DISK_MB)M $@.tmp && \
	 { mkfs.vfat $(if $(SOURCE_DATE_EPOCH),--invariant) -F 32 -n RINGZERO $@.tmp >/dev/null || mformat -F -v RINGZERO -i $@.tmp ::; } && mv -n $@.tmp $@ && \
	 echo "created $@ ($(DISK_MB) MiB FAT32)"

# M5b: WASI programs for /bin, built or fetched by tools/wasm_apps.sh into WASM_DIR, then copied onto the disk
# with mtools. Nothing is formatted: other files stay, same-name files in /bin are replaced. Refuses while a
# guest has the image open (stop it first: !poweroff).
# M8: the common set (sqlite3, jq, lua, python3, kibi next to coreutils, rg, qjs) and CPython's stdlib in
# /lib/python3.15 (files are replaced, the __pycache__ that python3 writes on the disk stays), and the inputs of
# docs/DEMOS.md in /demo (tools/demo). J = build threads.
WASM_DIR ?= $(BUILD)/wasm
J ?= 12
wasm-apps: $(DISK)
	@if pgrep -f "file=$(DISK),[i]f=none" >/dev/null; then echo "refusing: a guest has $(DISK) open"; exit 1; fi
	J=$(J) tools/wasm_apps.sh $(WASM_DIR)
	mmd -i $(DISK) ::/bin 2>/dev/null || true
	mcopy -o -i $(DISK) $(WASM_DIR)/*.wasm ::/bin/
	mmd -i $(DISK) ::/lib 2>/dev/null || true
	mcopy -o -s -i $(DISK) $(WASM_DIR)/lib/python3.15 ::/lib/
	mmd -i $(DISK) ::/demo 2>/dev/null || true
	mcopy -o -i $(DISK) tools/demo/* ::/demo/
	mdir -i $(DISK) ::/bin

# T34 beta (docs/BETA.md): both ISOs (4B q4, the recommended edition, and the default 2B q8) and one fresh data disk
# with the M8 apps and /demo in build/beta, plus SHA256SUMS. SOURCE_DATE_EPOCH (default: the last commit's time)
# fixes every timestamp xorriso (--set_all_file_dates), mkfs.vfat (--invariant: fixed volume id) and mtools write and
# the ISO's MBR disk id, so a second run on the
# same tree and inputs gives the same bytes. Needs the models (make model, both editions) and KVM (manual vectors;
# the 2B goes first so pass 1 boots a 6G guest).
BETA ?= $(BUILD)/beta
beta: export SOURCE_DATE_EPOCH ?= $(shell git log -1 --format=%ct 2>/dev/null || echo 1764201600)
beta:
	mkdir -p $(BETA)
	$(MAKE) iso MODEL=2b QUANT=q8 ISO=$(BETA)/ringzero-beta-2b-q8.iso
	$(MAKE) iso MODEL=4b QUANT=q4 ISO=$(BETA)/ringzero-beta-4b-q4.iso
	rm -f $(BETA)/disk.img
	$(MAKE) wasm-apps DISK=$(BETA)/disk.img
	cd $(BETA) && sha256sum ringzero-beta-4b-q4.iso ringzero-beta-2b-q8.iso disk.img | tee SHA256SUMS

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
VNC_PASS   ?= ringzero
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
