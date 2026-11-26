#pragma once
#ifdef NDEBUG
#define assert(e) ((void)0)
#else
void __assert_func(const char *file, int line, const char *func, const char *expr) __attribute__((noreturn));
#define assert(e) ((e) ? (void)0 : __assert_func(__FILE__, __LINE__, __func__, #e))
#endif
