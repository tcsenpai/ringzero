; enable_avx_fpu: copied from github.com/tcsenpai/llm.asm, branch qwen35-kernel, commit 1c73255
; (02-baremetal-os/boot.asm). Must run on EVERY core before it executes engine code.
; Clears CR0.EM, sets CR0.MP, sets CR4.OSFXSR|OSXMMEXCPT|OSXSAVE, sets XCR0 |= 7 (x87|SSE|AVX),
; then fninit and MXCSR = 0x1F80 (round-to-nearest-even, all exceptions masked).

bits 64
default rel

section .text
global enable_avx_fpu

enable_avx_fpu:
    mov     rax, cr0
    and     ax, 0xFFFB                  ; clear CR0.EM (bit 2)
    or      ax, 0x0002                  ; set CR0.MP (bit 1)
    mov     cr0, rax

    mov     rax, cr4
    or      eax, (1 << 9) | (1 << 10) | (1 << 18) ; OSFXSR | OSXMMEXCPT | OSXSAVE
    mov     cr4, rax

    xor     ecx, ecx                    ; XCR0
    xgetbv
    or      eax, 7                      ; x87 (bit 0) | SSE (bit 1) | AVX (bit 2)
    xsetbv

    fninit
    ldmxcsr [rel default_mxcsr]
    ret

section .rodata
align 4
default_mxcsr: dd 0x1F80
