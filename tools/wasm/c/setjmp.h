// M8: setjmp/longjmp for wasm32-wasip1 without the exception-handling proposal (wasmi has none, and wasi-libc's
// own setjmp.h needs it). Binaryen's Asyncify does the jump: setjmp unwinds the wasm stack to main() and rewinds
// it at once, keeping the saved locals in jmp_buf; longjmp unwinds and rewinds from that copy. The idea and the
// state machine follow Ruby's wasm/setjmp.c. The module must go through `wasm-opt --asyncify` (rzsjlj.c has main).
// This header shadows the sysroot's <setjmp.h> (-I before the sysroot).
#ifndef RZ_SETJMP_H
#define RZ_SETJMP_H

// saved wasm locals of every frame from main() down to setjmp; a deeper stack traps in Asyncify (unreachable)
#define RZ_JMP_BYTES 16384

struct rz_abuf {
    void *top, *end; // Asyncify's data header: next free byte, end of the buffer
    char data[RZ_JMP_BYTES];
};
typedef struct {
    struct rz_abuf a;
    void *top; // a.top after the capture: each rewind pops from it, so it is restored before the next one
    int val, state;
} rz_jmp[1];
typedef rz_jmp jmp_buf;

int rz_setjmp(rz_jmp env) __attribute__((noinline));
void rz_longjmp(rz_jmp env, int val) __attribute__((noinline));

// longjmp returns to Asyncify's unwind check right after the call, never to the next C statement
#define setjmp(env) ((env)->state = 0, rz_setjmp(env))
#define longjmp(env, v) (rz_longjmp(env, v), __builtin_unreachable())
#define _setjmp setjmp
#define _longjmp longjmp

#endif
