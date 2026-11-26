// The few libc functions MicroPython needs beyond Rust's compiler_builtins (memcpy, memmove, memset, memcmp).
#include <stddef.h>
#include <stdint.h>

void upy_fatal(const char *msg) __attribute__((noreturn));

int errno;

size_t strlen(const char *s) {
    const char *p = s;
    while (*p) {
        p++;
    }
    return p - s;
}

int strcmp(const char *a, const char *b) {
    while (*a && *a == *b) {
        a++, b++;
    }
    return (unsigned char)*a - (unsigned char)*b;
}

int strncmp(const char *a, const char *b, size_t n) {
    for (; n; n--, a++, b++) {
        if (*a != *b || !*a) {
            return (unsigned char)*a - (unsigned char)*b;
        }
    }
    return 0;
}

char *strchr(const char *s, int c) {
    for (;; s++) {
        if (*s == (char)c) {
            return (char *)s;
        }
        if (!*s) {
            return NULL;
        }
    }
}

void *memchr(const void *s, int c, size_t n) {
    const unsigned char *p = s;
    for (; n; n--, p++) {
        if (*p == (unsigned char)c) {
            return (void *)p;
        }
    }
    return NULL;
}

double fabs(double x) {
    union { double f; uint64_t i; } u = { x };
    u.i &= ~(1ull << 63);
    return u.f;
}

double nan(const char *tag) {
    (void)tag;
    return __builtin_nan("");
}

void abort(void) {
    upy_fatal("MicroPython: abort()");
}
