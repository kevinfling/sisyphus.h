/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

/* example_denoise_2d.c
 * Demonstrate spectral denoising of a 2-D grid via separable DCT-II.
 */
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    int nx = 64, ny = 64;
    int dims[2] = {nx, ny};
    size_t total = (size_t)nx * ny;
    double* grid = (double*)malloc(total * sizeof(double));

    srand((unsigned)time(NULL));
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double x = (double)i / (nx - 1);
            double y = (double)j / (ny - 1);
            double clean = sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
            double noise = 0.2 * ((double)rand() / RAND_MAX - 0.5);
            grid[i * ny + j] = clean + noise;
        }
    }

    int ret = sis_denoise_nd(2, dims, grid, 0.08, NULL);
    if (ret != SIS_OK) {
        fprintf(stderr, "denoise_nd failed: %d\n", ret);
        return 1;
    }

    printf("2D Denoising example (%dx%d, cutoff=0.08)\n", nx, ny);
    printf("  Sample output at center row (i=%d):\n", nx / 2);
    int i = nx / 2;
    for (int j = 0; j < ny; j += 8) {
        printf("    j=%d: %+.4f\n", j, grid[i * ny + j]);
    }

    free(grid);
    sis_fftw_cleanup();
    return 0;
}
