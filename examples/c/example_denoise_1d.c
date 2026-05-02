/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

/* example_denoise_1d.c
 * Demonstrate spectral denoising of a 1-D signal via DCT-II low-pass filter.
 */
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    int n = 256;
    double* signal = (double*)malloc(n * sizeof(double));
    double* noisy  = (double*)malloc(n * sizeof(double));

    srand((unsigned)time(NULL));
    for (int i = 0; i < n; ++i) {
        double t = (double)i / (n - 1);
        signal[i] = sin(2.0 * M_PI * 3.0 * t);
        noisy[i]  = signal[i] + 0.3 * ((double)rand() / RAND_MAX - 0.5);
    }

    /* Denoise: keep only lowest 5%% of DCT modes */
    int ret = sis_denoise_1d(noisy, n, 0.05);
    if (ret != SIS_OK) {
        fprintf(stderr, "denoise failed: %d\n", ret);
        return 1;
    }

    double mse_after = 0.0;
    for (int i = 0; i < n; ++i) {
        double db = noisy[i] - signal[i];
        /* We overwrote noisy with denoised; to compute before MSE we would
         * need a copy. For brevity we just print the denoised signal. */
        mse_after += db * db;
    }
    mse_after /= n;
    printf("1D Denoising example (n=%d, cutoff=0.05)\n", n);
    printf("  MSE after denoising: %.6f\n", mse_after);
    printf("  t, original, denoised\n");
    for (int i = 0; i < n; i += 16) {
        printf("  %.4f  %+.6f  %+.6f\n", (double)i / (n - 1), signal[i], noisy[i]);
    }

    free(signal); free(noisy);
    sis_fftw_cleanup();
    return 0;
}
