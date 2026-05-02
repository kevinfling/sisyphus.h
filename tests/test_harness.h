/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <math.h>

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_ASSERT(cond) do { \
    g_tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } else { \
        g_tests_passed++; \
    } \
} while(0)

#define TEST_ASSERT_NEAR(a, b, tol) do { \
    g_tests_run++; \
    double _a = (a), _b = (b), _tol = (tol); \
    if (fabs(_a - _b) > _tol) { \
        fprintf(stderr, "  FAIL: %s:%d: |%g - %g| = %g > %g\n", \
                __FILE__, __LINE__, _a, _b, fabs(_a - _b), _tol); \
        return 1; \
    } else { \
        g_tests_passed++; \
    } \
} while(0)

static inline int test_report(const char* name) {
    if (g_tests_run == g_tests_passed) {
        printf("[PASS] %s (%d/%d)\n", name, g_tests_passed, g_tests_run);
        return 0;
    } else {
        printf("[FAIL] %s (%d/%d)\n", name, g_tests_passed, g_tests_run);
        return 1;
    }
}

#endif
