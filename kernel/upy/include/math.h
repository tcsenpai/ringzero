// Declarations for lib/libm_dbl (musl) + gcc builtins for the classification macros.
#pragma once
typedef float float_t;
typedef double double_t;
#define NAN (__builtin_nan(""))
#define INFINITY (__builtin_inf())
#define HUGE_VAL (__builtin_huge_val())
#define FP_NAN 0
#define FP_INFINITE 1
#define FP_ZERO 2
#define FP_SUBNORMAL 3
#define FP_NORMAL 4
#define M_E 2.7182818284590452354
#define M_PI 3.14159265358979323846
#define isnan(x) __builtin_isnan(x)
#define isinf(x) __builtin_isinf(x)
#define isfinite(x) __builtin_isfinite(x)
#define signbit(x) __builtin_signbit(x)
#define fpclassify(x) __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, x)
double fabs(double), sqrt(double), pow(double, double), exp(double), expm1(double), log(double), log2(double);
double log10(double), log1p(double), cosh(double), sinh(double), tanh(double), acosh(double), asinh(double);
double atanh(double), cos(double), sin(double), tan(double), acos(double), asin(double), atan(double);
double atan2(double, double), ceil(double), copysign(double, double), floor(double), fmod(double, double);
double frexp(double, int *), ldexp(double, int), modf(double, double *), trunc(double), erf(double);
double erfc(double), tgamma(double), lgamma(double), nearbyint(double), round(double), rint(double);
double scalbn(double, int), nan(const char *);
