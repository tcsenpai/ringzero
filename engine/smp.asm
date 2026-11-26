; smp.asm: SMP thread-pool dispatch for the engine (tp_matmul_q8_0, tp_gqa_attn, tp_deltanet) and the
; AP worker loop. Adapted from llm.asm by Alexey Fateev (alesha-pro), https://github.com/alesha-pro/llm.asm (MIT); taken via our fork github.com/tcsenpai/llm.asm, branch qwen35-kernel, commit 1c73255 (adds Qwen3.5)
; (02-baremetal-os/smp.asm, MIT). Changes from upstream:
;   - AP bring-up (MADT, trampoline, INIT-SIPI-SIPI) is gone: Limine MP starts the APs and jumps to
;     ap_limine_entry with the limine_mp_info pointer in rdi (tid in its extra_argument, +24).
;   - parking: an idle worker spins for tp_park_tsc TSC cycles, then (if tp_park_enable) sleeps in
;     hlt. The dispatcher sends one broadcast IPI (x2APIC, vector WAKE_VECTOR) when tp_n_parked != 0.
;   - tp_shutdown removed (nothing calls it in the Rust kernel).
; The partitioning and the job protocol (tp_gen / tp_done_arr) are unchanged, so results stay
; bit-identical across core counts.

bits 64
default rel

%include "constants.inc"

%ifndef FAST_Q8
%define FAST_Q8 1
%endif

%define OS_MAX_CPUS     64
%define AP_STACK_SIZE   65536
%define WAKE_VECTOR     0xF0
%define X2APIC_ICR      0x830
%define X2APIC_EOI      0x80B

extern enable_avx_fpu
extern ap_init                          ; Rust: void ap_init(u64 tid) (lidt + LAPIC enable)
extern matmul_q8_0_range
%if FAST_Q8
extern quantize_x_q8_0
extern gqa_attn_range
%ifdef QWEN35
extern deltanet_heads_range
%endif
%endif

section .text

global ap_limine_entry
global tp_matmul_q8_0
%if FAST_Q8
global tp_gqa_attn
%ifdef QWEN35
global tp_deltanet
%endif
%endif
global lapic_eoi_isr
global tp_n_threads
global tp_n_online
global tp_n_parked
global tp_park_enable
global tp_park_tsc
global tp_wakeups

; ----------------------------------------------------------------------------
; AP entry from Limine MP: rdi = struct limine_mp_info * (extra_argument = tid at +24).
; ----------------------------------------------------------------------------
ap_limine_entry:
    cli
    mov     r12, [rdi + 24]             ; r12 = my worker tid (1..)
    cmp     r12, OS_MAX_CPUS
    jae     ap_halt

    ; private 64 KiB stack: slot tid-1, top = ap_stacks + tid * AP_STACK_SIZE (16B aligned)
    mov     rax, r12
    shl     rax, 16
    lea     rsp, [rel ap_stacks]
    add     rsp, rax

    mov     rdi, r12
    call    ap_init
    call    enable_avx_fpu
    lock inc dword [rel tp_n_online]

    mov     r13d, [rel tp_gen]          ; my_gen
.worker_wait:
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    add     rax, [rel tp_park_tsc]
    mov     rbx, rax                    ; rbx = park deadline (TSC)
.spin:
    mov     eax, [rel tp_gen]
    cmp     eax, r13d
    jne     .got_job
    pause
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    cmp     rax, rbx
    jb      .spin
    cmp     dword [rel tp_park_enable], 0
    je      .spin
    ; park: announce, re-check (pairs with the dispatcher's lock xadd tp_gen + load tp_n_parked)
    lock inc dword [rel tp_n_parked]
    mov     eax, [rel tp_gen]
    cmp     eax, r13d
    jne     .unpark
    sti                                 ; sti;hlt has no interrupt window in between
    hlt
    cli
.unpark:
    lock dec dword [rel tp_n_parked]
    jmp     .worker_wait

.got_job:
    mov     r13d, eax                   ; latch new generation

    ; Only participate if tid < tp_n_threads
    mov     r8, [rel tp_n_threads]
    cmp     r12, r8
    jae     .job_done

    ; Deterministic balanced row/head slice [r0, r1) for tid = r12
    mov     rcx, [rel tp_M]
%if FAST_Q8
    lea     r9, [r8 - 1]                ; N - 1
    mov     rax, rcx
    imul    rax, r12
    add     rax, r9
    xor     edx, edx
    div     r8
    mov     r14, rax                    ; r0

    lea     rax, [r12 + 1]
    cmp     rax, r8
    je      .last_slice
    imul    rax, rcx
    add     rax, r9
    xor     edx, edx
    div     r8
    mov     r15, rax                    ; r1
    jmp     .run_slice
%else
    mov     r9, rcx
    shr     r9, 2                       ; blocks4 = M / 4

    mov     rax, r9
    imul    rax, r12
    xor     edx, edx
    div     r8
    shl     rax, 2
    mov     r14, rax                    ; r0

    lea     rax, [r12 + 1]
    cmp     rax, r8
    je      .last_slice
    imul    rax, r9
    xor     edx, edx
    div     r8
    shl     rax, 2
    mov     r15, rax                    ; r1
    jmp     .run_slice
%endif
.last_slice:
    mov     r15, rcx                    ; r1 = M

.run_slice:
    cmp     r15, r14
    jle     .job_done

    mov     rdi, [rel tp_W]
    mov     rsi, [rel tp_x]
    mov     rdx, [rel tp_out]
    mov     rcx, [rel tp_K]
%if FAST_Q8
    test    rcx, rcx
    jz      .run_attn
%ifdef QWEN35
    cmp     rcx, 1
    je      .run_dn
%endif
%endif
    mov     r8,  r14                    ; r0
    mov     r9,  r15                    ; r1
    call    matmul_q8_0_range
    jmp     .job_done
%if FAST_Q8
.run_attn:
    mov     rcx, r14                    ; h0
    mov     r8,  r15                    ; h1
    call    gqa_attn_range
%ifdef QWEN35
    jmp     .job_done
.run_dn:                                ; K == 1: Gated-DeltaNet head job
    mov     rdi, r14                    ; h0
    mov     rsi, r15                    ; h1
    call    deltanet_heads_range
%endif
%endif

.job_done:
    ; Lock-free TSO release store to this core's private 64B cache line
    lea     rax, [rel tp_done_arr]
    mov     rcx, r12
    shl     rcx, 6                      ; tid * 64
    mov     [rax + rcx], r13d
    jmp     .worker_wait

ap_halt:
    cli
    hlt
    jmp     ap_halt

; ----------------------------------------------------------------------------
; EOI-only handler for the wake IPI and the BSP idle timer (x2APIC). Touches rax/rcx/rdx only.
; ----------------------------------------------------------------------------
lapic_eoi_isr:
    push    rax
    push    rcx
    push    rdx
    mov     ecx, X2APIC_EOI
    xor     eax, eax
    xor     edx, edx
    wrmsr
    pop     rdx
    pop     rcx
    pop     rax
    iretq

; ----------------------------------------------------------------------------
; void tp_matmul_q8_0(const void *W, const float *x, float *out, uint64 M, uint64 K)
; Exact same ABI and 4-row-aligned partitioning as Phase 1 src/threads.asm!
; ----------------------------------------------------------------------------
tp_matmul_q8_0:
%if FAST_Q8
    ; Quantize input vector x[0..K-1] to Q8_0 ONCE before waking AP cores
    push    rdi
    push    rsi
    push    rdx
    push    rcx
    push    r8
    mov     rdi, rsi                    ; x
    mov     rsi, r8                     ; K
    call    quantize_x_q8_0
    pop     r8
    pop     rcx
    pop     rdx
    pop     rsi
    pop     rdi
%endif
    mov     rax, [rel tp_n_threads]
    cmp     rax, 1
    jle     .single_thread

    push    rbx
    push    r12
    sub     rsp, 8                      ; 16B aligned
    mov     rbx, rax                    ; n_threads

    mov     [rel tp_W], rdi
    mov     [rel tp_x], rsi
    mov     [rel tp_out], rdx
    mov     [rel tp_M], rcx
    mov     [rel tp_K], r8

    ; Full memory barrier + broadcast new generation to all spinning APs
    mov     r12d, 1
    lock xadd dword [rel tp_gen], r12d
    inc     r12d                        ; r12d = new_gen

.run_bsp_slice:
    ; Wake parked workers (one broadcast IPI). The lock xadd above orders this load.
    cmp     dword [rel tp_n_parked], 0
    je      .no_wake
    mov     ecx, X2APIC_ICR
    mov     eax, 0x000C0000 | WAKE_VECTOR ; all-excluding-self, fixed, edge
    xor     edx, edx
    wrmsr
    inc     qword [rel tp_wakeups]
.no_wake:
    ; Run BSP's own slice (tid = 0)
    mov     rax, [rel tp_M]
%if FAST_Q8
    lea     rax, [rax + rbx - 1]
    xor     edx, edx
    div     rbx                         ; r1 for tid = 0: ceil(M / N)
%else
    shr     rax, 2                      ; blocks4
    xor     edx, edx
    div     rbx
    shl     rax, 2                      ; r1 for tid = 0
%endif
    test    rax, rax
    jz      .wait_workers

    mov     rdi, [rel tp_W]
    mov     rsi, [rel tp_x]
    mov     rdx, [rel tp_out]
    mov     rcx, [rel tp_K]
%if FAST_Q8
    test    rcx, rcx
    jz      .bsp_attn
%ifdef QWEN35
    cmp     rcx, 1
    je      .bsp_dn
%endif
%endif
    xor     r8d, r8d                    ; r0 = 0
    mov     r9,  rax                    ; r1
    call    matmul_q8_0_range
    jmp     .wait_workers
%if FAST_Q8
.bsp_attn:
    xor     ecx, ecx                    ; h0 = 0
    mov     r8,  rax                    ; h1
    call    gqa_attn_range
%ifdef QWEN35
    jmp     .wait_workers
.bsp_dn:
    xor     edi, edi                    ; h0 = 0
    mov     rsi, rax                    ; h1
    call    deltanet_heads_range
%endif
%endif

.wait_workers:
    ; Spin-wait for each worker tid = 1 .. n_threads - 1
    lea     rdi, [rel tp_done_arr]
    mov     rcx, 1
.wait_tid:
    cmp     rcx, rbx
    jae     .all_done
    mov     rdx, rcx
    shl     rdx, 6
.spin_tid:
    cmp     dword [rdi + rdx], r12d
    je      .next_tid
    pause
    jmp     .spin_tid
.next_tid:
    inc     rcx
    jmp     .wait_tid

.all_done:
    lfence
    add     rsp, 8
    pop     r12
    pop     rbx
    ret

.single_thread:
    mov     r9,  rcx                    ; r1 = M
    mov     rcx, r8                     ; K
    xor     r8d, r8d                    ; r0 = 0
    jmp     matmul_q8_0_range

%if FAST_Q8
; ----------------------------------------------------------------------------
; void tp_gqa_attn(uint64 layer_kv_base, uint64 pos, uint64 scale_gqa_packed,
;                  uint64 n_heads)
; Dispatches GQA attention across SMP AP cores (tp_K = 0 signals attn job).
; ----------------------------------------------------------------------------
tp_gqa_attn:
    mov     rax, [rel tp_n_threads]
    cmp     rax, 1
    jle     .attn_single

    push    rbx
    push    r12
    sub     rsp, 8
    mov     rbx, rax

    mov     [rel tp_W], rdi
    mov     [rel tp_x], rsi
    mov     [rel tp_out], rdx
    mov     [rel tp_M], rcx
    mov     qword [rel tp_K], 0

    mov     r12d, 1
    lock xadd dword [rel tp_gen], r12d
    inc     r12d
    jmp     tp_matmul_q8_0.run_bsp_slice

.attn_single:
    mov     r8, rcx                     ; h1 = n_heads
    xor     ecx, ecx                    ; h0 = 0
    jmp     gqa_attn_range
%ifdef QWEN35

; ----------------------------------------------------------------------------
; void tp_deltanet(uint64 n_heads)   job kind tp_K = 1; params are in transformer.asm g_dn_job
; ----------------------------------------------------------------------------
tp_deltanet:
    mov     rax, [rel tp_n_threads]
    cmp     rax, 1
    jle     .dn_single
    push    rbx
    push    r12
    sub     rsp, 8
    mov     rbx, rax
    mov     [rel tp_M], rdi
    mov     qword [rel tp_K], 1
    mov     r12d, 1
    lock xadd dword [rel tp_gen], r12d
    inc     r12d
    jmp     tp_matmul_q8_0.run_bsp_slice
.dn_single:
    mov     rsi, rdi                    ; h1 = n_heads
    xor     edi, edi                    ; h0 = 0
    jmp     deltanet_heads_range
%endif
%endif

; ============================================================================
; Thread-pool state (.bss)
; ============================================================================
section .bss
alignb 64
tp_gen:            resd 1
tp_pad0:           resd 1
tp_n_threads:      resq 1
tp_W:              resq 1
tp_x:              resq 1
tp_out:            resq 1
tp_M:              resq 1
tp_K:              resq 1

alignb 64
tp_n_parked:       resd 1               ; own line: written by parking workers
alignb 64
tp_park_enable:    resd 1               ; read-mostly (Rust writes between requests)
tp_n_online:       resd 1
tp_park_tsc:       resq 1
alignb 64
tp_wakeups:        resq 1               ; BSP-only counter, kept off the workers' lines

alignb 64
tp_done_arr:       resb OS_MAX_CPUS * 64

alignb 64
ap_stacks:         resb OS_MAX_CPUS * AP_STACK_SIZE
