// MicroPython configuration for ringzero (ports/embed, freestanding x86-64, ring 0).
// Feature level EXTRA (f-strings, set/frozenset ops, ...) minus everything that needs an OS:
// no filesystem import, no sys.stdin/stdout objects, no threads. Floats are double (lib/libm_dbl).
#include <stdint.h>

typedef long mp_off_t;
#define alloca(n) __builtin_alloca(n)
#define SSIZE_MAX __LONG_MAX__
#define MICROPY_MPHALPORT_H "upy_hal.h"

#define MICROPY_CONFIG_ROM_LEVEL                (MICROPY_CONFIG_ROM_LEVEL_EXTRA_FEATURES)

#define MICROPY_ENABLE_COMPILER                 (1)
#define MICROPY_ENABLE_GC                       (1)
#define MICROPY_HELPER_REPL                     (0)
#define MICROPY_ENABLE_EXTERNAL_IMPORT          (0)
#define MICROPY_READER_VFS                      (0)
#define MICROPY_PY_SYS_STDFILES                 (0)
#define MICROPY_PY_SYS_PLATFORM                 "ringzero"
#define MICROPY_PY_IO                           (0)
#define MICROPY_PY_BUILTINS_INPUT               (0)
#define MICROPY_PY_UCTYPES                      (0)
#define MICROPY_PY_THREAD                       (0)
#define MICROPY_PERSISTENT_CODE_LOAD            (0)
#define MICROPY_ENABLE_FINALISER                (0)
#define MICROPY_ERROR_REPORTING                 (MICROPY_ERROR_REPORTING_DETAILED)
#define MICROPY_ENABLE_SOURCE_LINE              (1)
#define MICROPY_LONGINT_IMPL                    (MICROPY_LONGINT_IMPL_MPZ)
#define MICROPY_FLOAT_IMPL                      (MICROPY_FLOAT_IMPL_DOUBLE)
#define MICROPY_PY_MATH                         (1)
#define MICROPY_PY_CMATH                        (1)
#define MICROPY_STACK_CHECK                     (1)
#define MICROPY_USE_INTERNAL_ERRNO              (1)

// Ctrl-C and the time budget (upy.rs): the VM calls upy_poll every MICROPY_VM_HOOK_COUNT
// backward jumps / returns; it raises KeyboardInterrupt via mp_sched_keyboard_interrupt
// or aborts the run outright (glue.c).
#define MICROPY_KBD_EXCEPTION                   (1)
#define MICROPY_ENABLE_SCHEDULER                (1)
void upy_poll(void);
#define MICROPY_VM_HOOK_COUNT                   (256)
#define MICROPY_VM_HOOK_INIT                    static unsigned int vm_hook_divisor = MICROPY_VM_HOOK_COUNT;
#define MICROPY_VM_HOOK_POLL                    if (--vm_hook_divisor == 0) { vm_hook_divisor = MICROPY_VM_HOOK_COUNT; upy_poll(); }
#define MICROPY_VM_HOOK_LOOP                    MICROPY_VM_HOOK_POLL
#define MICROPY_VM_HOOK_RETURN                  MICROPY_VM_HOOK_POLL
