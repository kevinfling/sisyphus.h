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

static int test_jacobian_pwc(void) {
    int n = 6;
    int partition[6] = {0, 0, 1, 1, 2, 2};
    double grad_in[3] = {1.0, 2.0, 3.0};
    double grad_out[6];

    int ret = sis_weak_jacobian_pwc(n, partition, 3, grad_in, grad_out);
    TEST_ASSERT(ret == SIS_OK);

    TEST_ASSERT_NEAR(grad_out[0], 1.0, 1e-12);
    TEST_ASSERT_NEAR(grad_out[1], 1.0, 1e-12);
    TEST_ASSERT_NEAR(grad_out[2], 2.0, 1e-12);
    TEST_ASSERT_NEAR(grad_out[3], 2.0, 1e-12);
    TEST_ASSERT_NEAR(grad_out[4], 3.0, 1e-12);
    TEST_ASSERT_NEAR(grad_out[5], 3.0, 1e-12);

    return 0;
}

static int test_jacobian_pwl(void) {
    int n = 3;
    double x[3] = {0.0, 0.5, 1.0};
    double t[3] = {0.0, 0.5, 1.0};
    double grad_in[3] = {0.0, 1.0, 2.0};
    double grad_out[3];

    int ret = sis_weak_jacobian_pwl(n, x, t, 3, grad_in, grad_out);
    TEST_ASSERT(ret == SIS_OK);

    /* x[0]=0.0 -> interval 0, weight (0.5-0.0)/0.5 = 1.0 for t[0] */
    TEST_ASSERT_NEAR(grad_out[0], 0.0, 1e-12);
    /* x[1]=0.5 -> interval 1 (last), weight 0 for t[1] since h=0.5, (1.0-0.5)/0.5=1.0 for t[2]? Wait, x[1]=0.5 is exactly t[1], so interval j=1, but loop breaks at j=0 since x[i]>=t[0] && x[i]<t[1] is false (0.5<0.5 false). Then j increments... Actually j will be 1 because 0.5>=0.5 && 0.5<1.0 true. Then h=0.5, a=(1.0-0.5)/0.5=1.0, b=0.0. So grad_out[1] = 1.0*grad_in[1] + 0.0*grad_in[2] = 1.0. */
    TEST_ASSERT_NEAR(grad_out[1], 1.0, 1e-12);
    /* x[2]=1.0 -> loop: j=0 false, j=1 true (1.0>=0.5 && 1.0<1.0 false). Loop ends with j=2 (k-1). h = t[2]-t[1] = 0.5. a = (1.0-1.0)/0.5 = 0, b = (1.0-0.5)/0.5 = 1.0. grad_out[2] = 0*grad_in[1] + 1.0*grad_in[2] = 2.0. */
    TEST_ASSERT_NEAR(grad_out[2], 2.0, 1e-12);

    return 0;
}

int main(void) {
    int failures = 0;
    g_tests_run = 0; g_tests_passed = 0;
    failures += test_jacobian_pwc();
    failures += test_report("test_jacobian_pwc");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_jacobian_pwl();
    failures += test_report("test_jacobian_pwl");

    sis_fftw_cleanup();
    return failures;
}
