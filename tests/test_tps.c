/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include "test_harness.h"
#include <math.h>
#include <stdlib.h>

static int test_tps_2d_fit(void) {
    int n = 25;
    double xi[50]; /* 25 * 2 */
    double yi[25];
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            int idx = i * 5 + j;
            xi[idx * 2 + 0] = (double)i / 4.0;
            xi[idx * 2 + 1] = (double)j / 4.0;
            double x = xi[idx * 2 + 0];
            double y = xi[idx * 2 + 1];
            yi[idx] = x * x - y * y; /* saddle */
        }
    }

    sis_tps_t s;
    int ret = sis_tps_fit(&s, xi, yi, n, 2, 1e-6);
    TEST_ASSERT(ret == SIS_OK);

    for (int i = 0; i < n; ++i) {
        double xq[2] = {xi[i * 2 + 0], xi[i * 2 + 1]};
        double v = sis_tps_eval(&s, xq);
        TEST_ASSERT_NEAR(v, yi[i], 0.01);
    }

    sis_tps_free(&s);
    return 0;
}

int main(void) {
    int failures = 0;
    g_tests_run = 0; g_tests_passed = 0;
    failures += test_tps_2d_fit();
    failures += test_report("test_tps_2d_fit");

    sis_fftw_cleanup();
    return failures;
}
