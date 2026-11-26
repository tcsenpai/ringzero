// M8: Asyncify setjmp/longjmp (see setjmp.h). The program's main is renamed rz_main (-Dmain=rz_main); this main
// calls it in a loop: when it comes back because of an unwind (a setjmp capture or a longjmp), the loop stops
// the unwind and rewinds into the setjmp call. rz_setjmp, rz_longjmp and main call the asyncify.* builtins, so
// Asyncify does not instrument them: rz_setjmp is called again by its rewound caller and returns 0 or the value.
#include "setjmp.h"
#include <stdlib.h>

__attribute__((import_module("asyncify"), import_name("start_unwind"))) void as_start_unwind(void *);
__attribute__((import_module("asyncify"), import_name("stop_unwind"))) void as_stop_unwind(void);
__attribute__((import_module("asyncify"), import_name("start_rewind"))) void as_start_rewind(void *);
__attribute__((import_module("asyncify"), import_name("stop_rewind"))) void as_stop_rewind(void);

enum { FRESH, CAPTURING, CAPTURED, RETURNING };

static struct rz_abuf scratch; // longjmp's unwind: written, never rewound
static rz_jmp *active_env;      // the jmp_buf of the unwind in flight
static void *unwinding;         // its Asyncify buffer, NULL when no unwind is in flight
#define ACTIVE (*active_env)

// inlined: the asyncify.* calls must sit in rz_setjmp / rz_longjmp themselves
static inline __attribute__((always_inline)) void unwind(struct rz_abuf *b) {
    b->top = b->data;
    b->end = b->data + RZ_JMP_BYTES;
    unwinding = b;
    as_start_unwind(b);
}

int rz_setjmp(rz_jmp env) {
    switch (env->state) {
    case FRESH:
        env->state = CAPTURING;
        active_env = (rz_jmp *)env;
        unwind(&env->a);
        return -1; // not seen: the caller unwinds
    case CAPTURING: // rewound after the capture
        as_stop_rewind();
        env->state = CAPTURED;
        return 0;
    case RETURNING: // rewound by longjmp
        as_stop_rewind();
        env->state = CAPTURED;
        return env->val;
    }
    abort();
}

void rz_longjmp(rz_jmp env, int val) {
    if (env->state != CAPTURED) abort();
    env->state = RETURNING;
    env->val = val ? val : 1;
    active_env = (rz_jmp *)env;
    unwind(&scratch);
}

extern int rz_main(int argc, char **argv);

int main(int argc, char **argv) {
    for (;;) {
        int r = rz_main(argc, argv);
        if (!unwinding) return r;
        as_stop_unwind();
        unwinding = NULL;
        if (ACTIVE->state == CAPTURING)
            ACTIVE->top = ACTIVE->a.top;
        else
            ACTIVE->a.top = ACTIVE->top;
        as_start_rewind(&ACTIVE->a);
    }
}
