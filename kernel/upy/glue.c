// ringzero <-> MicroPython glue (replaces ports/embed's embed_util.c and mphalport.c).
// Each run gets a fresh interpreter (gc_init + mp_init ... mp_deinit) on its own stack, carved from the
// engine arena by upy.rs, so a Python program never eats into the 64 KiB boot stack. Imports from Rust:
// upy_out (stdout sink) and upy_check (time budget / Ctrl-C: 0 go on, 1 raise KeyboardInterrupt,
// 2 abort the run now, for programs that swallow KeyboardInterrupt).
#include <string.h>
#include "py/compile.h"
#include "py/gc.h"
#include "py/runtime.h"
#include "py/stackctrl.h"
#include "py/mphal.h"
#include "py/mperrno.h"
#include "shared/runtime/gchelper.h"

void upy_out(const char *s, size_t n);
int upy_check(void);
void upy_fatal(const char *msg) __attribute__((noreturn));

static char *heap_lo, *stack_lo;
static size_t heap_size, stack_size;
static void *abort_jb[5]; // __builtin_setjmp buffer: hard stop of a run that ignores KeyboardInterrupt

void upy_setup(void *heap, size_t hsz, void *stack, size_t ssz) {
    heap_lo = heap;
    heap_size = hsz;
    stack_lo = stack;
    stack_size = ssz;
}

// void upy_on_stack(void (*fn)(void *), void *arg, void *top): call fn(arg) with rsp = top (16-aligned).
__asm__(
    ".text\n"
    ".global upy_on_stack\n"
    "upy_on_stack:\n"
    "  push %rbp\n"
    "  mov %rsp, %rbp\n"
    "  mov %rdx, %rsp\n"
    "  mov %rdi, %rax\n"
    "  mov %rsi, %rdi\n"
    "  call *%rax\n"
    "  mov %rbp, %rsp\n"
    "  pop %rbp\n"
    "  ret\n");
void upy_on_stack(void (*fn)(void *), void *arg, void *top);

struct job {
    const char *src, *name;
    size_t len, name_len;
    int single; // one line: REPL-style input, the value of an expression is printed
    int rc;     // 0 ok, 1 uncaught exception (traceback printed), 2 aborted by upy_check, 3 fatal
    size_t used;
};

// mp_obj_print_exception with the traceback cut to its first and last TB_KEEP frames: a runaway recursion
// has thousands, printed from C where the time budget cannot reach.
#define TB_KEEP 5
static void print_exc(mp_obj_t exc) {
    const mp_print_t *print = &mp_plat_print;
    if (mp_obj_is_exception_instance(exc)) {
        size_t n, *values;
        mp_obj_exception_get_traceback(exc, &n, &values);
        size_t frames = n / 3;
        if (frames > 0) {
            mp_print_str(print, "Traceback (most recent call last):\n");
        }
        for (size_t p = 0; p < frames; p++) { // p = 0: outermost frame
            if (frames > 2 * TB_KEEP && p == TB_KEEP) {
                mp_printf(print, "  [... %u frames not shown ...]\n", (unsigned)(frames - 2 * TB_KEEP));
                p = frames - TB_KEEP - 1;
                continue;
            }
            size_t i = n - 3 - 3 * p;
            mp_printf(print, "  File \"%q\", line %d", values[i], (int)values[i + 1]);
            if (values[i + 2] == MP_QSTRnull) {
                mp_print_str(print, "\n");
            } else {
                mp_printf(print, ", in %q\n", values[i + 2]);
            }
        }
    }
    mp_obj_print_helper(print, exc, PRINT_EXC);
    mp_print_str(print, "\n");
}

static void run(void *arg) {
    struct job *j = arg;
    mp_stack_set_top(stack_lo + stack_size);
    mp_stack_set_limit(stack_size - 32 * 1024);
    gc_init(heap_lo, heap_lo + heap_size);
    mp_init();
    if (__builtin_setjmp(abort_jb) == 0) {
        nlr_buf_t nlr;
        if (nlr_push(&nlr) == 0) {
            qstr name = qstr_from_strn(j->name, j->name_len);
            mp_lexer_t *lex = mp_lexer_new_from_str_len(name, j->src, j->len, 0);
            mp_parse_tree_t pt = mp_parse(lex, j->single ? MP_PARSE_SINGLE_INPUT : MP_PARSE_FILE_INPUT);
            mp_obj_t f = mp_compile(&pt, name, j->single);
            mp_call_function_0(f);
            nlr_pop();
            j->rc = 0;
        } else {
            print_exc((mp_obj_t)nlr.ret_val);
            j->rc = 1;
        }
    } else {
        MP_STATE_THREAD(nlr_top) = NULL; // the abort skipped every nlr_pop
    }
    gc_info_t info;
    gc_info(&info);
    j->used = info.used;
    mp_deinit();
}

// Run Python source. Returns the rc above; *used = GC heap bytes in use at the end of the run.
int upy_exec(const char *src, size_t len, const char *name, size_t name_len, int single, size_t *used) {
    struct job j = { src, name, len, name_len, single, 2, 0 };
    upy_on_stack(run, &j, (void *)((uintptr_t)(stack_lo + stack_size) & ~(uintptr_t)15));
    *used = j.used;
    return j.rc;
}

void upy_poll(void) {
    switch (upy_check()) {
        case 1:
            mp_sched_keyboard_interrupt();
            break;
        case 2:
            __builtin_longjmp(abort_jb, 1);
    }
}

void gc_collect(void) {
    gc_collect_start();
    gc_helper_collect_regs_and_stack();
    gc_collect_end();
}

void mp_hal_stdout_tx_strn_cooked(const char *s, size_t n) {
    upy_out(s, n);
}

mp_uint_t mp_hal_stdout_tx_strn(const char *s, size_t n) {
    upy_out(s, n);
    return n;
}

// no files to import or execfile() from (yet)
mp_lexer_t *mp_lexer_new_from_file(qstr filename) {
    (void)filename;
    mp_raise_OSError(MP_ENOENT);
}

void nlr_jump_fail(void *val) {
    (void)val;
    upy_fatal("MicroPython: nlr_jump with no handler");
}
