/**===========================================================================
 * sisyphus.h
 * Zero-dependency (except FFTW3) header-only spectral toolbox — SISYPHUS
 * DCT-II (REDFT10) + Pepin-style high-degree splines + denoising + ND
 *
 * Compile: -std=c11 -O3 -march=native -lfftw3
 *
 * Mathematical foundation:
 *
 * 1. DCT-II (even extension) eliminates Gibbs ringing at non-periodic boundaries.
 *    We use FFTW3 REDFT10/REDFT01 exclusively for all transforms.
 *
 * 2. Pepin high-degree splines adapted to DCT:
 *    Instead of the DFT Toeplitz-Hessenberg system (which requires periodic
 *    data and suffers from Gibbs), we compute exact derivatives of the DCT-II
 *    interpolant via spectral differentiation in the cosine/sine basis.
 *    The piecewise Taylor polynomial of degree θ built from these derivatives
 *    is C^{θ-1} by construction, parameter-free, and stable for any N.
 *
 * 3. Multi-D:
 *    - Separable DCT for tensor-product grids.
 *    - Mixed partials via successive 1D spectral differentiation.
 *    - Tensor-product Pepin evaluators (exact C^{θ-1} in each axis).
 *    - Thin-plate splines (2-D/3-D polyharmonic RBF) for scattered data.
 *
 * 4. Smoothing splines via Demmler-Reinsch spectral shrinkage in DCT basis.
 *
 * 5. Differentiable spline layers (Cho et al.):
 *    Block-diagonal weak Jacobians for piecewise constant / linear backprop.
 *
 * Version:    v0.1.0
 * License:    MIT License (see end of file)
 * Copyright:  (c) 2026 Kevin Fling
 * Repository: https://github.com/kevinfling/sisyphus.h
 *
 * ===========================================================================*/

#ifndef SISYPHUS_H
#define SISYPHUS_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --- configuration --- */
#ifndef SIS_MAX_DIM
#define SIS_MAX_DIM 4
#endif
#ifndef SIS_MAX_DEGREE
#define SIS_MAX_DEGREE 11
#endif
#ifndef SIS_FFTW_CACHE_SIZE
#define SIS_FFTW_CACHE_SIZE 32
#endif

/* --- error codes --- */
enum {
    SIS_OK = 0,
    SIS_ERR_NOMEM = -1,
    SIS_ERR_SIZE = -2,
    SIS_ERR_FFTW = -3,
    SIS_ERR_ARG = -4,
    SIS_ERR_MATH = -5
};

/* --- 1D DCT (FFTW3 only) --- */
int sis_dct2(const double* in, double* out, int n);
int sis_idct2(const double* in, double* out, int n);

/* --- ND separable DCT (1D/2D/3D) --- */
int sis_dct2_nd(int ndim, const int* dims, const double* in, double* out, double* work);
int sis_idct2_nd(int ndim, const int* dims, const double* in, double* out, double* work);

/* --- Non-Uniform Fast Cosine Transform (NUFCT) --- */
/* Evaluate the DCT-II interpolant at non-uniform points xq[0..m-1].
 * L = physical domain length.  oversample <= 0 selects automatic (default 8). */
int sis_nufct_eval(int n, const double* y, int m, const double* xq,
                  double* out, double L, int oversample);

/* --- spectral denoising --- */
int sis_denoise_1d(double* inout, int n, double cutoff_cycles_per_knot);
int sis_denoise_nd(int ndim, const int* dims, double* inout, double cutoff_cycles_per_knot, double* work);

/* --- spectral differentiation (exact derivatives of DCT interpolant) --- */
/* mu-th derivative at all knots. scale = 1.0 for unit parametric domain [0,pi]. */
int sis_derivative_1d(const double* in, double* out, int n, int mu, double scale);
/* directional derivative along axis `dir` (0-based) */
int sis_derivative_nd(int ndim, const int* dims, const double* in, double* out,
                     int dir, int mu, double scale, double* work);
/* fractional derivative of order alpha > 0.  scale = pi / L. */
int sis_fractional_derivative_1d(const double* in, double* out, int n,
                                double alpha, double scale);

/* --- Barycentric rational interpolation (Floater-Hormann) --- */
typedef struct {
    int n;          /* number of knots - 1 */
    int d;          /* blending degree */
    double* x;      /* knots, size n+1 */
    double* f;      /* values, size n+1 */
    double* w;      /* barycentric weights, size n+1 */
} sis_barycentric_t;

int    sis_barycentric_init (sis_barycentric_t* s, int npoints, int degree);
void   sis_barycentric_free (sis_barycentric_t* s);
int    sis_barycentric_build(sis_barycentric_t* s, const double* x,
                            const double* y, int n, int degree);
double sis_barycentric_eval (const sis_barycentric_t* s, double xq);
int    sis_barycentric_fit_denoised(sis_barycentric_t* s, const double* x,
                                   const double* y, int n, int degree,
                                   double cutoff_cycles);

/* --- Pepin high-degree spline (1D) --- */
typedef struct {
    int n;          /* number of intervals (knots-1) */
    int degree;     /* theta */
    int uniform;    /* non-zero if x grid is uniform */
    double* x;      /* knots, size n+1 */
    double* c;      /* Taylor coeffs [interval][degree+1], row-major.
                       c[j][mu] = g^{(mu)}(x_j)  (undivided by factorial) */
} sis_pepin1d_t;

int  sis_pepin1d_init(sis_pepin1d_t* s, int npoints, int degree);
void sis_pepin1d_free(sis_pepin1d_t* s);
/* Build from uniform knots x[0..n-1] and values y[0..n-1].
 * The DCT interpolant is differentiated spectrally; scale = physical_length / M_PI. */
int  sis_pepin1d_build(sis_pepin1d_t* s, const double* x, const double* y, int n, int degree);
double sis_pepin1d_eval(const sis_pepin1d_t* s, double xq, int deriv);
double sis_pepin1d_integrate(const sis_pepin1d_t* s, double a, double b);

/* --- Tensor product spline (ND rectilinear grid) --- */
typedef struct {
    int ndim;
    int degree;
    int n[SIS_MAX_DIM];               /* points per dim */
    double* xk[SIS_MAX_DIM];          /* knot coordinates per dim */
    double* mixed;                   /* mixed partials at each grid point,
                                        size prod(n_i) * (degree+1)^ndim  */
    size_t mixed_stride;             /* (degree+1)^ndim */
} sis_tensor_t;

int    sis_tensor_init(sis_tensor_t* s, int ndim, const int* n, int degree);
void   sis_tensor_free(sis_tensor_t* s);
/* Build from rectilinear grid. xk[dim] is array of n[dim] coords.
 * y is row-major, size prod(n_i). */
int    sis_tensor_build(sis_tensor_t* s, const double* xk[SIS_MAX_DIM],
                       const double* y, int degree);
double sis_tensor_eval(const sis_tensor_t* s, const double* xq, int deriv_dir);

/* --- Thin plate spline (2-D / 3-D polyharmonic) --- */
typedef struct {
    int n;          /* centers */
    int d;          /* 2 or 3 */
    double* xi;     /* centers, n*d row-major */
    double* beta;   /* n RBF coeffs */
    double* poly;   /* d+1 polynomial coeffs (affine part) */
} sis_tps_t;

int    sis_tps_init(sis_tps_t* s, int n, int d);
void   sis_tps_free(sis_tps_t* s);
/* Fit TPS to scattered data. lambda > 0 for regularization. */
int    sis_tps_fit(sis_tps_t* s, const double* xi, const double* yi,
                  int n, int d, double lambda);
double sis_tps_eval(const sis_tps_t* s, const double* xq);

/* --- Smoothing spline (1-D natural, spectral Demmler-Reinsch) --- */
/* order = spline degree = 2*m-1.  m=2 => cubic.  lambda >= 0. */
int sis_smoothing_spline_1d(const double* x, const double* y, int n,
                           int order, double lambda,
                           double* out_smooth);

/* --- Differentiable spline layer (Cho et al. weak Jacobian) --- */
/* Piecewise constant (degree 0) backward pass.
 * partition[i] in [0,k-1] is the interval index of sample i.
 * grad_in[k] is upstream grad w.r.t. interval means. */
int sis_weak_jacobian_pwc(int n, const int* partition, int k,
                         const double* grad_in, double* grad_out);
/* Piecewise linear (degree 1) on knots t[0..k] (breakpoints, not intervals).
 * grad_in[k] is upstream grad w.r.t. knot values. */
int sis_weak_jacobian_pwl(int n, const double* x, const double* t, int k,
                         const double* grad_in, double* grad_out);

/* --- Convenience wrappers --- */
/* Denoise then fit a Pepin spline of given degree. */
int sis_pepin1d_fit_denoised(sis_pepin1d_t* s, const double* x, const double* y,
                            int n, int degree, double cutoff_cycles);

/* --- lifecycle --- */
void sis_fftw_init(void);   /* optional; called automatically */
void sis_fftw_cleanup(void);

/*============================================================================
 * IMPLEMENTATION
 *============================================================================*/
#ifdef SISYPHUS_IMPLEMENTATION

/* --- small helpers --- */
#define SIS_INLINE static inline __attribute__((always_inline))

SIS_INLINE double sis__ipow(double x, int e) {
    double r = 1.0;
    while (e > 0) { if (e & 1) r *= x; x *= x; e >>= 1; }
    return r;
}

/* factorials 0! .. 12! */
static const double sis__fact[13] = {
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0,
    40320.0, 362880.0, 3628800.0, 39916800.0, 479001600.0
};

/* --- FFTW plan cache (single-threaded; init from one thread) --- */
typedef struct {
    int n;
    fftw_plan dct2;      /* REDFT10 */
    fftw_plan idct2;     /* REDFT01 */
    fftw_plan idst2;     /* RODFT01  (for odd derivatives) */
    double* buf;         /* aligned scratch, size n */
} sis__plan_t;

static sis__plan_t sis__plans[SIS_FFTW_CACHE_SIZE];
static int sis__nplans = 0;

static int sis__find_plan(int n) {
    for (int i = 0; i < sis__nplans; ++i) if (sis__plans[i].n == n) return i;
    return -1;
}

static int sis__ensure_plan(int n) {
    int idx = sis__find_plan(n);
    if (idx >= 0) return idx;
    if (sis__nplans >= SIS_FFTW_CACHE_SIZE) return -1;
    idx = sis__nplans++;
    sis__plans[idx].n = n;
    sis__plans[idx].buf = (double*)fftw_malloc(sizeof(double) * n);
    sis__plans[idx].dct2  = fftw_plan_r2r_1d(n, sis__plans[idx].buf, sis__plans[idx].buf,
                                            FFTW_REDFT10, FFTW_ESTIMATE);
    sis__plans[idx].idct2 = fftw_plan_r2r_1d(n, sis__plans[idx].buf, sis__plans[idx].buf,
                                            FFTW_REDFT01, FFTW_ESTIMATE);
    sis__plans[idx].idst2 = fftw_plan_r2r_1d(n, sis__plans[idx].buf, sis__plans[idx].buf,
                                            FFTW_RODFT01, FFTW_ESTIMATE);
    return idx;
}

void sis_fftw_init(void) { sis__nplans = 0; }
void sis_fftw_cleanup(void) {
    for (int i = 0; i < sis__nplans; ++i) {
        fftw_destroy_plan(sis__plans[i].dct2);
        fftw_destroy_plan(sis__plans[i].idct2);
        fftw_destroy_plan(sis__plans[i].idst2);
        fftw_free(sis__plans[i].buf);
    }
    sis__nplans = 0;
}

/* --- 1D DCT --- */
int sis_dct2(const double* in, double* out, int n) {
    if (n < 1) return SIS_ERR_SIZE;
    int idx = sis__ensure_plan(n);
    if (idx < 0) return SIS_ERR_NOMEM;
    memcpy(sis__plans[idx].buf, in, n * sizeof(double));
    fftw_execute_r2r(sis__plans[idx].dct2, sis__plans[idx].buf, sis__plans[idx].buf);
    memcpy(out, sis__plans[idx].buf, n * sizeof(double));
    return SIS_OK;
}

int sis_idct2(const double* in, double* out, int n) {
    if (n < 1) return SIS_ERR_SIZE;
    int idx = sis__ensure_plan(n);
    if (idx < 0) return SIS_ERR_NOMEM;
    memcpy(sis__plans[idx].buf, in, n * sizeof(double));
    fftw_execute_r2r(sis__plans[idx].idct2, sis__plans[idx].buf, sis__plans[idx].buf);
    double norm = 1.0 / (2.0 * n);
    for (int i = 0; i < n; ++i) out[i] = sis__plans[idx].buf[i] * norm;
    return SIS_OK;
}

/* --- ND DCT (separable, up to 3D) --- */
static size_t sis__prod(const int* a, int n) {
    size_t p = 1;
    for (int i = 0; i < n; ++i) p *= (size_t)a[i];
    return p;
}

int sis_dct2_nd(int ndim, const int* dims, const double* in, double* out, double* work) {
    (void)work;
    if (ndim < 1 || ndim > 3) return SIS_ERR_ARG;
    size_t total = sis__prod(dims, ndim);
    memcpy(out, in, total * sizeof(double));

    for (int d = 0; d < ndim; ++d) {
        int n = dims[d];
        int idx = sis__ensure_plan(n);
        if (idx < 0) return SIS_ERR_NOMEM;
        fftw_plan plan = sis__plans[idx].dct2;
        double* buf = sis__plans[idx].buf;

        if (ndim == 1) {
            memcpy(buf, out, n * sizeof(double));
            fftw_execute_r2r(plan, buf, buf);
            memcpy(out, buf, n * sizeof(double));
        } else if (ndim == 2) {
            int stride = (d == 0) ? dims[1] : 1;
            int lines  = (d == 0) ? dims[1] : dims[0];
            for (int i = 0; i < lines; ++i) {
                for (int j = 0; j < n; ++j) buf[j] = out[i * stride + j * (stride==1?dims[1]:1)];
                fftw_execute_r2r(plan, buf, buf);
                for (int j = 0; j < n; ++j) out[i * stride + j * (stride==1?dims[1]:1)] = buf[j];
            }
        } else if (ndim == 3) {
            int s0 = dims[1] * dims[2];
            int s1 = dims[2];
            int s2 = 1;
            if (d == 0) {
                for (int y = 0; y < dims[1]; ++y)
                    for (int z = 0; z < dims[2]; ++z) {
                        for (int x = 0; x < n; ++x) buf[x] = out[x*s0 + y*s1 + z*s2];
                        fftw_execute_r2r(plan, buf, buf);
                        for (int x = 0; x < n; ++x) out[x*s0 + y*s1 + z*s2] = buf[x];
                    }
            } else if (d == 1) {
                for (int x = 0; x < dims[0]; ++x)
                    for (int z = 0; z < dims[2]; ++z) {
                        for (int y = 0; y < n; ++y) buf[y] = out[x*s0 + y*s1 + z*s2];
                        fftw_execute_r2r(plan, buf, buf);
                        for (int y = 0; y < n; ++y) out[x*s0 + y*s1 + z*s2] = buf[y];
                    }
            } else {
                for (int x = 0; x < dims[0]; ++x)
                    for (int y = 0; y < dims[1]; ++y) {
                        for (int z = 0; z < n; ++z) buf[z] = out[x*s0 + y*s1 + z*s2];
                        fftw_execute_r2r(plan, buf, buf);
                        for (int z = 0; z < n; ++z) out[x*s0 + y*s1 + z*s2] = buf[z];
                    }
            }
        }
    }
    return SIS_OK;
}

int sis_idct2_nd(int ndim, const int* dims, const double* in, double* out, double* work) {
    (void)work;
    if (ndim < 1 || ndim > 3) return SIS_ERR_ARG;
    size_t total = sis__prod(dims, ndim);
    memcpy(out, in, total * sizeof(double));

    for (int d = 0; d < ndim; ++d) {
        int n = dims[d];
        int idx = sis__ensure_plan(n);
        if (idx < 0) return SIS_ERR_NOMEM;
        fftw_plan plan = sis__plans[idx].idct2;
        double* buf = sis__plans[idx].buf;
        double norm = 1.0 / (2.0 * n);

        if (ndim == 1) {
            memcpy(buf, out, n * sizeof(double));
            fftw_execute_r2r(plan, buf, buf);
            for (int i = 0; i < n; ++i) out[i] = buf[i] * norm;
        } else if (ndim == 2) {
            int stride = (d == 0) ? dims[1] : 1;
            int lines  = (d == 0) ? dims[1] : dims[0];
            for (int i = 0; i < lines; ++i) {
                for (int j = 0; j < n; ++j) buf[j] = out[i * stride + j * (stride==1?dims[1]:1)];
                fftw_execute_r2r(plan, buf, buf);
                for (int j = 0; j < n; ++j) out[i * stride + j * (stride==1?dims[1]:1)] = buf[j] * norm;
            }
        } else if (ndim == 3) {
            int s0 = dims[1] * dims[2];
            int s1 = dims[2];
            int s2 = 1;
            if (d == 0) {
                for (int y = 0; y < dims[1]; ++y)
                    for (int z = 0; z < dims[2]; ++z) {
                        for (int x = 0; x < n; ++x) buf[x] = out[x*s0 + y*s1 + z*s2];
                        fftw_execute_r2r(plan, buf, buf);
                        for (int x = 0; x < n; ++x) out[x*s0 + y*s1 + z*s2] = buf[x] * norm;
                    }
            } else if (d == 1) {
                for (int x = 0; x < dims[0]; ++x)
                    for (int z = 0; z < dims[2]; ++z) {
                        for (int y = 0; y < n; ++y) buf[y] = out[x*s0 + y*s1 + z*s2];
                        fftw_execute_r2r(plan, buf, buf);
                        for (int y = 0; y < n; ++y) out[x*s0 + y*s1 + z*s2] = buf[y] * norm;
                    }
            } else {
                for (int x = 0; x < dims[0]; ++x)
                    for (int y = 0; y < dims[1]; ++y) {
                        for (int z = 0; z < n; ++z) buf[z] = out[x*s0 + y*s1 + z*s2];
                        fftw_execute_r2r(plan, buf, buf);
                        for (int z = 0; z < n; ++z) out[x*s0 + y*s1 + z*s2] = buf[z] * norm;
                    }
            }
        }
    }
    return SIS_OK;
}

/* --- spectral denoising --- */
int sis_denoise_1d(double* inout, int n, double cutoff_cycles) {
    if (n < 2) return SIS_ERR_SIZE;
    double* c = (double*)fftw_malloc(sizeof(double) * n);
    if (!c) return SIS_ERR_NOMEM;
    int ret = sis_dct2(inout, c, n);
    if (ret != SIS_OK) { fftw_free(c); return ret; }
    int cut = (int)(cutoff_cycles * n + 0.5);
    if (cut < 1) cut = 1;
    for (int k = cut; k < n; ++k) c[k] = 0.0;
    ret = sis_idct2(c, inout, n);
    fftw_free(c);
    return ret;
}

int sis_denoise_nd(int ndim, const int* dims, double* inout, double cutoff_cycles, double* work) {
    if (ndim < 1 || ndim > 3) return SIS_ERR_ARG;
    size_t total = sis__prod(dims, ndim);
    double* c = work ? work : (double*)fftw_malloc(sizeof(double) * total);
    if (!c) return SIS_ERR_NOMEM;
    int ret = sis_dct2_nd(ndim, dims, inout, c, NULL);
    if (ret == SIS_OK) {
        /* hyper-rectangular lowpass: zero if any normalized freq exceeds cutoff */
        if (ndim == 1) {
            int cut = (int)(cutoff_cycles * dims[0] + 0.5);
            for (int i = cut; i < dims[0]; ++i) c[i] = 0.0;
        } else if (ndim == 2) {
            for (int i = 0; i < dims[0]; ++i)
                for (int j = 0; j < dims[1]; ++j)
                    if (i > cutoff_cycles * dims[0] || j > cutoff_cycles * dims[1])
                        c[i * dims[1] + j] = 0.0;
        } else {
            for (int i = 0; i < dims[0]; ++i)
                for (int j = 0; j < dims[1]; ++j)
                    for (int k = 0; k < dims[2]; ++k)
                        if (i > cutoff_cycles * dims[0] || j > cutoff_cycles * dims[1] || k > cutoff_cycles * dims[2])
                            c[(i * dims[1] + j) * dims[2] + k] = 0.0;
        }
        ret = sis_idct2_nd(ndim, dims, c, inout, NULL);
    }
    if (!work) fftw_free(c);
    return ret;
}

/* --- spectral differentiation --- */
int sis_derivative_1d(const double* in, double* out, int n, int mu, double scale) {
    if (n < 2 || mu < 0) return SIS_ERR_ARG;
    if (mu == 0) { memcpy(out, in, n * sizeof(double)); return SIS_OK; }

    double* c = (double*)fftw_malloc(sizeof(double) * n);
    if (!c) return SIS_ERR_NOMEM;
    int ret = sis_dct2(in, c, n);
    if (ret != SIS_OK) { fftw_free(c); return ret; }

    /* convert to a_k coefficients of cosine series on [0,pi] */
    double inv_n = 1.0 / (double)n;
    c[0] *= 0.5 * inv_n;
    for (int k = 1; k < n; ++k) c[k] *= inv_n;

    int idx = sis__ensure_plan(n);
    if (idx < 0) { fftw_free(c); return SIS_ERR_NOMEM; }
    double* buf = sis__plans[idx].buf;

    double sign = ((mu % 4 == 1) || (mu % 4 == 2)) ? -1.0 : 1.0;
    double scl = scale;

    if (mu % 2 == 0) {
        /* even derivative -> cosine series -> REDFT01 */
        buf[0] = 0.0;
        for (int k = 1; k < n; ++k) {
            double w = sign * sis__ipow((double)k * scl, mu) * c[k];
            buf[k] = w;
        }
        fftw_execute_r2r(sis__plans[idx].idct2, buf, buf);
        for (int i = 0; i < n; ++i) out[i] = buf[i] * 0.5;
    } else {
        /* odd derivative -> sine series -> RODFT01 */
        for (int k = 0; k < n - 1; ++k) {
            double kk = (double)(k + 1);
            buf[k] = sign * sis__ipow(kk * scl, mu) * c[k + 1];
        }
        buf[n - 1] = 0.0;
        fftw_execute_r2r(sis__plans[idx].idst2, buf, buf);
        for (int i = 0; i < n; ++i) out[i] = buf[i] * 0.5;
    }
    fftw_free(c);
    return SIS_OK;
}

int sis_fractional_derivative_1d(const double* in, double* out, int n,
                                double alpha, double scale) {
    if (n < 2 || alpha < 0) return SIS_ERR_ARG;
    if (alpha == 0.0) { memcpy(out, in, n * sizeof(double)); return SIS_OK; }

    /* For integer alphas, delegate to the exact integer path */
    double nearest = round(alpha);
    if (fabs(alpha - nearest) < 1e-12 && nearest <= INT_MAX && nearest >= 0) {
        return sis_derivative_1d(in, out, n, (int)nearest, scale);
    }

    double* c = (double*)fftw_malloc(sizeof(double) * n);
    if (!c) return SIS_ERR_NOMEM;
    int ret = sis_dct2(in, c, n);
    if (ret != SIS_OK) { fftw_free(c); return ret; }

    double inv_n = 1.0 / (double)n;
    c[0] *= 0.5 * inv_n;
    for (int k = 1; k < n; ++k) c[k] *= inv_n;

    int idx = sis__ensure_plan(n);
    if (idx < 0) { fftw_free(c); return SIS_ERR_NOMEM; }
    double* buf_cos = sis__plans[idx].buf;
    double* buf_sin = (double*)fftw_malloc(n * sizeof(double));
    if (!buf_sin) { fftw_free(c); return SIS_ERR_NOMEM; }

    double ca = cos(alpha * M_PI * 0.5);
    double sa = sin(alpha * M_PI * 0.5);
    double scl = pow(scale, alpha);

    buf_cos[0] = 0.0;
    for (int k = 1; k < n; ++k) {
        double w = ca * pow((double)k, alpha) * c[k];
        buf_cos[k] = w;
    }
    for (int k = 0; k < n - 1; ++k) {
        double w = sa * pow((double)(k + 1), alpha) * c[k + 1];
        buf_sin[k] = w;
    }
    buf_sin[n - 1] = 0.0;

    fftw_execute_r2r(sis__plans[idx].idct2, buf_cos, buf_cos);
    fftw_execute_r2r(sis__plans[idx].idst2, buf_sin, buf_sin);

    for (int i = 0; i < n; ++i) {
        out[i] = (buf_cos[i] - buf_sin[i]) * 0.5 * scl;
    }

    fftw_free(buf_sin);
    fftw_free(c);
    return SIS_OK;
}

int sis_derivative_nd(int ndim, const int* dims, const double* in, double* out,
                     int dir, int mu, double scale, double* work) {
    if (ndim < 1 || ndim > 3 || dir < 0 || dir >= ndim) return SIS_ERR_ARG;
    size_t total = sis__prod(dims, ndim);
    double* tmp = work ? work : (double*)fftw_malloc(sizeof(double) * total);
    if (!tmp) return SIS_ERR_NOMEM;
    memcpy(tmp, in, total * sizeof(double));

    int ret = SIS_OK;
    if (ndim == 1) {
        ret = sis_derivative_1d(tmp, out, dims[0], mu, scale);
    } else if (ndim == 2) {
        int n = dims[dir];
        int lines = (dir == 0) ? dims[1] : dims[0];
        int stride = (dir == 0) ? dims[1] : 1;
        double* pencil = (double*)fftw_malloc(n * sizeof(double));
        if (!pencil) { if (!work) fftw_free(tmp); return SIS_ERR_NOMEM; }
        for (int i = 0; i < lines; ++i) {
            for (int j = 0; j < n; ++j)
                pencil[j] = tmp[i * stride + j * (stride==1?dims[1]:1)];
            ret = sis_derivative_1d(pencil, pencil, n, mu, scale);
            if (ret != SIS_OK) break;
            for (int j = 0; j < n; ++j)
                out[i * stride + j * (stride==1?dims[1]:1)] = pencil[j];
        }
        fftw_free(pencil);
    } else {
        int n = dims[dir];
        int n0 = dims[0], n1 = dims[1], n2 = dims[2];
        double* pencil = (double*)fftw_malloc(n * sizeof(double));
        if (!pencil) { if (!work) fftw_free(tmp); return SIS_ERR_NOMEM; }
        if (dir == 0) {
            for (int y = 0; y < n1; ++y)
                for (int z = 0; z < n2; ++z) {
                    for (int x = 0; x < n; ++x) pencil[x] = tmp[(x*n1 + y)*n2 + z];
                    ret = sis_derivative_1d(pencil, pencil, n, mu, scale);
                    if (ret != SIS_OK) break;
                    for (int x = 0; x < n; ++x) out[(x*n1 + y)*n2 + z] = pencil[x];
                }
        } else if (dir == 1) {
            for (int x = 0; x < n0; ++x)
                for (int z = 0; z < n2; ++z) {
                    for (int y = 0; y < n; ++y) pencil[y] = tmp[(x*n1 + y)*n2 + z];
                    ret = sis_derivative_1d(pencil, pencil, n, mu, scale);
                    if (ret != SIS_OK) break;
                    for (int y = 0; y < n; ++y) out[(x*n1 + y)*n2 + z] = pencil[y];
                }
        } else {
            for (int x = 0; x < n0; ++x)
                for (int y = 0; y < n1; ++y) {
                    for (int z = 0; z < n; ++z) pencil[z] = tmp[(x*n1 + y)*n2 + z];
                    ret = sis_derivative_1d(pencil, pencil, n, mu, scale);
                    if (ret != SIS_OK) break;
                    for (int z = 0; z < n; ++z) out[(x*n1 + y)*n2 + z] = pencil[z];
                }
        }
        fftw_free(pencil);
    }

    if (!work) fftw_free(tmp);
    return ret;
}

/* --- Pepin 1D --- */
int sis_pepin1d_init(sis_pepin1d_t* s, int npoints, int degree) {
    memset(s, 0, sizeof(*s));
    if (npoints < 2 || degree < 1 || degree > SIS_MAX_DEGREE) return SIS_ERR_ARG;
    s->n = npoints - 1;
    s->degree = degree;
    s->x = (double*)malloc(npoints * sizeof(double));
    s->c = (double*)malloc((size_t)s->n * (degree + 1) * sizeof(double));
    if (!s->x || !s->c) { sis_pepin1d_free(s); return SIS_ERR_NOMEM; }
    return SIS_OK;
}

void sis_pepin1d_free(sis_pepin1d_t* s) {
    free(s->x); s->x = NULL;
    free(s->c); s->c = NULL;
    s->n = 0; s->degree = 0;
}

int sis_pepin1d_build(sis_pepin1d_t* s, const double* x, const double* y, int n, int degree) {
    int ret = sis_pepin1d_init(s, n, degree);
    if (ret != SIS_OK) return ret;
    memcpy(s->x, x, n * sizeof(double));
    /* detect uniform grid */
    s->uniform = 1;
    if (n > 2) {
        double h = s->x[1] - s->x[0];
        for (int i = 2; i < n; ++i) {
            if (fabs(s->x[i] - s->x[i-1] - h) > 1e-12 * fabs(h) + 1e-15) {
                s->uniform = 0; break;
            }
        }
    }

    double L = x[n - 1] - x[0];
    double scale = (L > 0) ? M_PI / L : 1.0;

    /* compute derivatives mu = 0..degree at all knots via spectral differentiation */
    double* dbuf = (double*)fftw_malloc(n * sizeof(double));
    if (!dbuf) { sis_pepin1d_free(s); return SIS_ERR_NOMEM; }

    memcpy(dbuf, y, n * sizeof(double));
    for (int mu = 0; mu <= degree; ++mu) {
        if (mu > 0) {
            ret = sis_derivative_1d(dbuf, dbuf, n, 1, scale);
            if (ret != SIS_OK) { fftw_free(dbuf); sis_pepin1d_free(s); return ret; }
        }
        for (int j = 0; j < n - 1; ++j) {
            s->c[j * (degree + 1) + mu] = dbuf[j];
        }
    }
    fftw_free(dbuf);
    return SIS_OK;
}

double sis_pepin1d_eval(const sis_pepin1d_t* s, double xq, int deriv) {
    if (!s->n || deriv > s->degree) return NAN;
    /* find interval */
    int j = 0;
    if (xq < s->x[0]) j = 0;
    else if (xq >= s->x[s->n]) j = s->n - 1;
    else if (s->uniform) {
        double h = s->x[1] - s->x[0];
        j = (int)((xq - s->x[0]) / h);
        if (j < 0) j = 0;
        else if (j >= s->n) j = s->n - 1;
    } else {
        int lo = 0, hi = s->n;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (xq < s->x[mid+1]) hi = mid;
            else lo = mid + 1;
        }
        j = lo;
    }
    double dx = xq - s->x[j];
    double* a = &s->c[j * (s->degree + 1)];
    /* Horner for Taylor series: sum_{k=0}^{deg-deriv} a_{k+deriv} * dx^k / k! */
    int dmax = s->degree;
    double res = 0.0;
    for (int k = dmax; k > deriv; --k) {
        res = res * dx / (double)(k - deriv) + a[k];
    }
    res = res * dx + a[deriv];
    return res;
}

double sis_pepin1d_integrate(const sis_pepin1d_t* s, double a, double b) {
    if (!s->n) return 0.0;
    if (a > b) { double t = a; a = b; b = t; }
    if (a < s->x[0]) a = s->x[0];
    if (b > s->x[s->n]) b = s->x[s->n];
    double sum = 0.0;
    for (int j = 0; j < s->n; ++j) {
        double l = s->x[j], r = s->x[j+1];
        if (r <= a || l >= b) continue;
        double L = (r < b ? r : b) - (l > a ? l : a);
        double* c = &s->c[j * (s->degree + 1)];
        double term = 0.0;
        for (int mu = 0; mu <= s->degree; ++mu) {
            term += c[mu] * sis__ipow(L, mu + 1) / (sis__fact[mu] * (mu + 1.0));
        }
        sum += term;
    }
    return sum;
}

/* --- Tensor product spline --- */
int sis_tensor_init(sis_tensor_t* s, int ndim, const int* n, int degree) {
    memset(s, 0, sizeof(*s));
    if (ndim < 1 || ndim > SIS_MAX_DIM) return SIS_ERR_ARG;
    if (degree < 1 || degree > SIS_MAX_DEGREE) return SIS_ERR_ARG;
    s->ndim = ndim;
    s->degree = degree;
    size_t total_n = 1;
    for (int d = 0; d < ndim; ++d) {
        if (n[d] < 2) return SIS_ERR_SIZE;
        s->n[d] = n[d];
        total_n *= (size_t)n[d];
    }
    s->mixed_stride = 1;
    for (int d = 0; d < ndim; ++d) s->mixed_stride *= (size_t)(degree + 1);

    size_t mixed_sz = total_n * s->mixed_stride;
    if (mixed_sz > (size_t)1024 * 1024 * 1024) return SIS_ERR_SIZE; /* sanity 1GB cap */
    s->mixed = (double*)malloc(mixed_sz * sizeof(double));
    if (!s->mixed) return SIS_ERR_NOMEM;

    for (int d = 0; d < ndim; ++d) {
        s->xk[d] = (double*)malloc(n[d] * sizeof(double));
        if (!s->xk[d]) { sis_tensor_free(s); return SIS_ERR_NOMEM; }
    }
    return SIS_OK;
}

void sis_tensor_free(sis_tensor_t* s) {
    free(s->mixed); s->mixed = NULL;
    for (int d = 0; d < SIS_MAX_DIM; ++d) { free(s->xk[d]); s->xk[d] = NULL; }
    memset(s, 0, sizeof(*s));
}

/* Build mixed partials via successive 1D spectral differentiation along each axis.
 * y is row-major, size prod(n_i). */
int sis_tensor_build(sis_tensor_t* s, const double* xk[SIS_MAX_DIM],
                    const double* y, int degree) {
    int ndim = s->ndim;
    int n[SIS_MAX_DIM];
    size_t total = 1;
    for (int d = 0; d < ndim; ++d) {
        n[d] = s->n[d];
        memcpy(s->xk[d], xk[d], n[d] * sizeof(double));
        total *= (size_t)n[d];
    }

    /* working array: current field values */
    double* f = (double*)fftw_malloc(total * sizeof(double));
    if (!f) return SIS_ERR_NOMEM;
    memcpy(f, y, total * sizeof(double));

    /* For each grid point we need all mixed partials up to `degree` in each dim.
     * We compute them by differentiating along each axis repeatedly.
     * mixed[point][alpha_1][alpha_2]... where alpha_d in [0,degree].
     * We fill by recursion over dimensions. */
    memset(s->mixed, 0, total * s->mixed_stride * sizeof(double));

    /* Helper: index of mixed partial alpha vector for a grid point */
    /* We'll compute via separable application: for each dimension, differentiate
     * the current set of functions. */

    /* For simplicity, implement recursive generation for up to 3D.
     * Start with alpha = (0,0,0) = f itself. */
    for (size_t i = 0; i < total; ++i) s->mixed[i * s->mixed_stride] = f[i];

    int max_alpha[SIS_MAX_DIM] = {0};
    for (int d = 0; d < ndim; ++d) max_alpha[d] = degree;

    /* Iterate over all derivative multi-indices alpha */
    for (int a0 = 0; a0 <= degree; ++a0)
     for (int a1 = 0; a1 <= (ndim>1?degree:0); ++a1)
      for (int a2 = 0; a2 <= (ndim>2?degree:0); ++a2) {
          if (a0==0 && a1==0 && a2==0) continue;
          /* determine which axis was incremented last */
          int dir = -1, prev_a[3] = {a0,a1,a2};
          if (a2 > 0 && ndim > 2) { dir = 2; prev_a[2]--; }
          else if (a1 > 0 && ndim > 1) { dir = 1; prev_a[1]--; }
          else { dir = 0; prev_a[0]--; }

          size_t dst_stride = 0;
          size_t prev_stride = 0;
          /* mixed_stride = (degree+1)^ndim */
          size_t pow_base = (size_t)(degree + 1);
          if (ndim == 1) {
              dst_stride = a0;
              prev_stride = prev_a[0];
          } else if (ndim == 2) {
              dst_stride = a0 * pow_base + a1;
              prev_stride = prev_a[0] * pow_base + prev_a[1];
          } else {
              dst_stride = ((a0 * pow_base) + a1) * pow_base + a2;
              prev_stride = ((prev_a[0] * pow_base) + prev_a[1]) * pow_base + prev_a[2];
          }

          /* restore parent partial into f before differentiating */
          for (size_t i = 0; i < total; ++i)
              f[i] = s->mixed[i * s->mixed_stride + prev_stride];

          double scale = M_PI / (s->xk[dir][n[dir]-1] - s->xk[dir][0]);
          /* differentiate prev field along dir */
          int ret = sis_derivative_nd(ndim, n, f, f, dir, 1, scale, NULL);
          if (ret != SIS_OK) { fftw_free(f); return ret; }
          /* store result into mixed at dst_stride */
          for (size_t i = 0; i < total; ++i)
              s->mixed[i * s->mixed_stride + dst_stride] = f[i];
      }

    fftw_free(f);
    return SIS_OK;
}

double sis_tensor_eval(const sis_tensor_t* s, const double* xq, int deriv_dir) {
    if (!s->mixed) return NAN;
    int nd = s->ndim;
    int deg = s->degree;

    /* locate cell */
    int idx[SIS_MAX_DIM] = {0};
    double dx[SIS_MAX_DIM] = {0};
    for (int d = 0; d < nd; ++d) {
        int n = s->n[d];
        const double* xs = s->xk[d];
        if (xq[d] <= xs[0]) { idx[d] = 0; dx[d] = xq[d] - xs[0]; }
        else if (xq[d] >= xs[n-1]) { idx[d] = n-2; dx[d] = xq[d] - xs[n-2]; }
        else {
            for (int i = 0; i < n-1; ++i) {
                if (xq[d] >= xs[i] && xq[d] < xs[i+1]) { idx[d] = i; dx[d] = xq[d] - xs[i]; break; }
            }
        }
    }

    /* grid point linear index */
    size_t gidx = 0;
    if (nd == 1) gidx = idx[0];
    else if (nd == 2) gidx = idx[0] * s->n[1] + idx[1];
    else gidx = (idx[0] * s->n[1] + idx[1]) * s->n[2] + idx[2];

    double* m = &s->mixed[gidx * s->mixed_stride];
    size_t pb = (size_t)(deg + 1);

    double sum = 0.0;
    /* Horner-style evaluation over all axes simultaneously is messy;
     * for moderate degree we just loop over all monomials. */
    if (nd == 1) {
        int d0 = (deriv_dir == 0) ? 1 : 0;
        for (int a0 = d0; a0 <= deg; ++a0) {
            double coeff = m[a0];
            double term = coeff * sis__ipow(dx[0], a0 - d0) / sis__fact[a0 - d0];
            sum += term;
        }
    } else if (nd == 2) {
        int d0 = (deriv_dir == 0) ? 1 : 0;
        int d1 = (deriv_dir == 1) ? 1 : 0;
        for (int a0 = d0; a0 <= deg; ++a0)
            for (int a1 = d1; a1 <= deg; ++a1) {
                double coeff = m[a0 * pb + a1];
                double term = coeff *
                            sis__ipow(dx[0], a0 - d0) / sis__fact[a0 - d0] *
                            sis__ipow(dx[1], a1 - d1) / sis__fact[a1 - d1];
                sum += term;
            }
    } else {
        int d0 = (deriv_dir == 0) ? 1 : 0;
        int d1 = (deriv_dir == 1) ? 1 : 0;
        int d2 = (deriv_dir == 2) ? 1 : 0;
        for (int a0 = d0; a0 <= deg; ++a0)
         for (int a1 = d1; a1 <= deg; ++a1)
          for (int a2 = d2; a2 <= deg; ++a2) {
              double coeff = m[((a0 * pb) + a1) * pb + a2];
              double term = coeff *
                          sis__ipow(dx[0], a0 - d0) / sis__fact[a0 - d0] *
                          sis__ipow(dx[1], a1 - d1) / sis__fact[a1 - d1] *
                          sis__ipow(dx[2], a2 - d2) / sis__fact[a2 - d2];
              sum += term;
          }
    }
    return sum;
}

/* --- Thin plate spline --- */
int sis_tps_init(sis_tps_t* s, int n, int d) {
    memset(s, 0, sizeof(*s));
    if ((d != 2 && d != 3) || n < 1) return SIS_ERR_ARG;
    s->n = n; s->d = d;
    s->xi = (double*)malloc((size_t)n * d * sizeof(double));
    s->beta = (double*)malloc((size_t)n * sizeof(double));
    s->poly = (double*)malloc((size_t)(d + 1) * sizeof(double));
    if (!s->xi || !s->beta || !s->poly) { sis_tps_free(s); return SIS_ERR_NOMEM; }
    return SIS_OK;
}

void sis_tps_free(sis_tps_t* s) {
    free(s->xi); s->xi = NULL;
    free(s->beta); s->beta = NULL;
    free(s->poly); s->poly = NULL;
    s->n = 0; s->d = 0;
}

static double sis__tps_kernel(int d, double r) {
    if (d == 2) return (r == 0.0) ? 0.0 : r * r * log(r);
    else return r; /* 3D polyharmonic r */
}

/* small dense LU with partial pivoting */
static int sis__lu_solve(int n, double* A, double* b) {
    int* piv = (int*)malloc(n * sizeof(int));
    if (!piv) return -1;
    for (int i = 0; i < n; ++i) piv[i] = i;

    for (int col = 0; col < n; ++col) {
        int maxr = col;
        double maxv = fabs(A[col * n + col]);
        for (int r = col + 1; r < n; ++r) {
            double v = fabs(A[r * n + col]);
            if (v > maxv) { maxv = v; maxr = r; }
        }
        if (maxv < 1e-15) { free(piv); return -1; }
        if (maxr != col) {
            for (int k = 0; k < n; ++k) {
                double t = A[col * n + k]; A[col * n + k] = A[maxr * n + k]; A[maxr * n + k] = t;
            }
            double t = b[col]; b[col] = b[maxr]; b[maxr] = t;
            int tt = piv[col]; piv[col] = piv[maxr]; piv[maxr] = tt;
        }
        for (int r = col + 1; r < n; ++r) {
            double f = A[r * n + col] / A[col * n + col];
            A[r * n + col] = 0.0;
            for (int k = col + 1; k < n; ++k) A[r * n + k] -= f * A[col * n + k];
            b[r] -= f * b[col];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int k = i + 1; k < n; ++k) sum -= A[i * n + k] * b[k];
        b[i] = sum / A[i * n + i];
    }
    free(piv);
    return 0;
}

int sis_tps_fit(sis_tps_t* s, const double* xi, const double* yi,
               int n, int d, double lambda) {
    int ret = sis_tps_init(s, n, d);
    if (ret != SIS_OK) return ret;
    memcpy(s->xi, xi, (size_t)n * d * sizeof(double));

    int m = n + d + 1;
    double* A = (double*)calloc((size_t)m * m, sizeof(double));
    double* rhs = (double*)calloc((size_t)m, sizeof(double));
    if (!A || !rhs) { free(A); free(rhs); sis_tps_free(s); return SIS_ERR_NOMEM; }

    /* K block */
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double r2 = 0.0;
            for (int k = 0; k < d; ++k) {
                double diff = xi[i * d + k] - xi[j * d + k];
                r2 += diff * diff;
            }
            double r = sqrt(r2);
            A[i * m + j] = sis__tps_kernel(d, r);
            if (i == j) A[i * m + j] += lambda;
        }
        /* polynomial block P */
        A[i * m + n] = 1.0;
        for (int k = 0; k < d; ++k) A[i * m + n + 1 + k] = xi[i * d + k];
        rhs[i] = yi[i];
    }
    /* transpose constraints */
    for (int j = 0; j < n; ++j) {
        A[n * m + j] = 1.0;
        for (int k = 0; k < d; ++k) A[(n + 1 + k) * m + j] = xi[j * d + k];
    }
    /* zero lower-right */
    for (int i = n; i < m; ++i) for (int j = n; j < m; ++j) A[i * m + j] = 0.0;
    rhs[n] = 0.0;
    for (int k = 0; k < d; ++k) rhs[n + 1 + k] = 0.0;

    ret = sis__lu_solve(m, A, rhs);
    if (ret != 0) { free(A); free(rhs); sis_tps_free(s); return SIS_ERR_MATH; }

    memcpy(s->beta, rhs, n * sizeof(double));
    for (int k = 0; k < d + 1; ++k) s->poly[k] = rhs[n + k];
    free(A); free(rhs);
    return SIS_OK;
}

double sis_tps_eval(const sis_tps_t* s, const double* xq) {
    if (!s->xi) return NAN;
    double sum = s->poly[0];
    for (int k = 0; k < s->d; ++k) sum += s->poly[1 + k] * xq[k];
    for (int i = 0; i < s->n; ++i) {
        double r2 = 0.0;
        for (int k = 0; k < s->d; ++k) {
            double diff = xq[k] - s->xi[i * s->d + k];
            r2 += diff * diff;
        }
        sum += s->beta[i] * sis__tps_kernel(s->d, sqrt(r2));
    }
    return sum;
}

/* --- Smoothing spline (1D natural, spectral filter) --- */
int sis_smoothing_spline_1d(const double* x, const double* y, int n,
                           int order, double lambda,
                           double* out_smooth) {
    if (n < 2 || order < 1 || lambda < 0) return SIS_ERR_ARG;
    int m = (order + 1) / 2; /* penalty order */
    double* c = (double*)fftw_malloc(n * sizeof(double));
    if (!c) return SIS_ERR_NOMEM;
    int ret = sis_dct2(y, c, n);
    if (ret != SIS_OK) { fftw_free(c); return ret; }

    double L = x[n - 1] - x[0];
    double scale = M_PI / L;
    for (int k = 0; k < n; ++k) {
        double w = 1.0 + lambda * sis__ipow((double)k * scale, 2 * m);
        c[k] /= w;
    }
    ret = sis_idct2(c, out_smooth, n);
    fftw_free(c);
    return ret;
}

/* --- Weak Jacobian (Cho et al.) --- */
int sis_weak_jacobian_pwc(int n, const int* partition, int k,
                         const double* grad_in, double* grad_out) {
    if (!partition || !grad_in || !grad_out) return SIS_ERR_ARG;
    /* piecewise constant: each output element x_i contributes equally to its block */
    for (int i = 0; i < n; ++i) {
        int b = partition[i];
        if (b < 0 || b >= k) return SIS_ERR_ARG;
        grad_out[i] = grad_in[b];
    }
    return SIS_OK;
}

int sis_weak_jacobian_pwl(int n, const double* x, const double* t, int k,
                         const double* grad_in, double* grad_out) {
    if (!x || !t || !grad_in || !grad_out) return SIS_ERR_ARG;
    for (int i = 0; i < n; ++i) {
        /* find interval in t containing x[i] */
        int j = 0;
        for (j = 0; j < k - 1; ++j) if (x[i] >= t[j] && x[i] < t[j+1]) break;
        if (j >= k - 1) j = k - 2; /* clamp to last interval */
        double h = t[j+1] - t[j];
        double a = (h > 0) ? (t[j+1] - x[i]) / h : 0.0;
        double b = (h > 0) ? (x[i] - t[j]) / h : 0.0;
        grad_out[i] = a * grad_in[j] + b * grad_in[j+1];
    }
    return SIS_OK;
}

/* --- Barycentric rational implementation --- */
int sis_barycentric_init(sis_barycentric_t* s, int npoints, int degree) {
    memset(s, 0, sizeof(*s));
    if (npoints < 2 || degree < 0) return SIS_ERR_ARG;
    s->n = npoints - 1;
    s->d = degree;
    s->x = (double*)malloc((size_t)npoints * sizeof(double));
    s->f = (double*)malloc((size_t)npoints * sizeof(double));
    s->w = (double*)malloc((size_t)npoints * sizeof(double));
    if (!s->x || !s->f || !s->w) { sis_barycentric_free(s); return SIS_ERR_NOMEM; }
    return SIS_OK;
}

void sis_barycentric_free(sis_barycentric_t* s) {
    free(s->x); s->x = NULL;
    free(s->f); s->f = NULL;
    free(s->w); s->w = NULL;
    s->n = 0; s->d = 0;
}

static double sis__binom(int d, int k) {
    if (k < 0 || k > d) return 0.0;
    double r = 1.0;
    for (int i = 1; i <= k; ++i) {
        r = r * (double)(d - k + i) / (double)i;
    }
    return r;
}

int sis_barycentric_build(sis_barycentric_t* s, const double* x,
                         const double* y, int n, int degree) {
    int ret = sis_barycentric_init(s, n, degree);
    if (ret != SIS_OK) return ret;
    memcpy(s->x, x, n * sizeof(double));
    memcpy(s->f, y, n * sizeof(double));
    int d = degree;

    for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        int i0 = (j > d) ? (j - d) : 0;
        int i1 = (j < n - d) ? j : (n - d - 1);
        for (int i = i0; i <= i1; ++i) {
            sum += sis__binom(d, j - i);
        }
        double sign = ((j - d) % 2 == 0) ? 1.0 : -1.0;
        s->w[j] = sign * sum;
    }
    return SIS_OK;
}

double sis_barycentric_eval(const sis_barycentric_t* s, double xq) {
    if (!s->w) return NAN;
    int n = s->n + 1;
    const double* x = s->x;
    const double* f = s->f;
    const double* w = s->w;

    /* exact knot match */
    for (int j = 0; j < n; ++j) {
        if (fabs(xq - x[j]) < 1e-14) return f[j];
    }

    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; ++j) {
        double t = w[j] / (xq - x[j]);
        num += t * f[j];
        den += t;
    }
    return (den != 0.0) ? (num / den) : NAN;
}

int sis_barycentric_fit_denoised(sis_barycentric_t* s, const double* x,
                                const double* y, int n, int degree,
                                double cutoff_cycles) {
    double* yc = (double*)fftw_malloc(n * sizeof(double));
    if (!yc) return SIS_ERR_NOMEM;
    memcpy(yc, y, n * sizeof(double));
    int ret = sis_denoise_1d(yc, n, cutoff_cycles);
    if (ret == SIS_OK) ret = sis_barycentric_build(s, x, yc, n, degree);
    fftw_free(yc);
    return ret;
}

/* --- NUFCT --- */
static double sis__nufct_fold(double x, double L) {
    if (L <= 0) return x;
    double p = 2.0 * L;
    x = fmod(x, p);
    if (x < 0) x += p;
    if (x > L) x = p - x;
    return x;
}

static double sis__nufct_interp_cubic(const double* g, int M, int j, double u) {
    /* 4-point Lagrange stencil on uniform grid, u in [0,1] */
    int im1 = j - 1, ip1 = j + 1, ip2 = j + 2;
    /* reflect at boundaries using even symmetry */
    if (im1 < 0) im1 = -im1 - 1;
    if (ip1 >= M) ip1 = 2 * M - 1 - ip1;
    if (ip2 >= M) ip2 = 2 * M - 1 - ip2;
    double ym1 = g[im1];
    double y0  = g[j];
    double y1  = g[ip1];
    double y2  = g[ip2];
    double u1 = u - 1.0, u2 = u - 2.0, up1 = u + 1.0;
    return ym1 * (-u * u1 * u2 / 6.0)
         + y0  * ( up1 * u1 * u2 / 2.0)
         + y1  * (-up1 * u  * u2 / 2.0)
         + y2  * ( up1 * u  * u1 / 6.0);
}

int sis_nufct_eval(int n, const double* y, int m, const double* xq,
                  double* out, double L, int oversample) {
    if (n < 2 || m < 1 || L <= 0) return SIS_ERR_ARG;
    if (oversample <= 0) oversample = 8;

    /* Direct summation for small problems */
    if ((size_t)n * (size_t)m < 10000) {
        double* c = (double*)fftw_malloc(n * sizeof(double));
        if (!c) return SIS_ERR_NOMEM;
        int ret = sis_dct2(y, c, n);
        if (ret != SIS_OK) { fftw_free(c); return ret; }
        double inv_n = 1.0 / (double)n;
        c[0] *= 0.5 * inv_n;
        for (int k = 1; k < n; ++k) c[k] *= inv_n;
        for (int i = 0; i < m; ++i) {
            double t = M_PI * sis__nufct_fold(xq[i], L) / L;
            double val = c[0];
            for (int k = 1; k < n; ++k) val += c[k] * cos((double)k * t);
            out[i] = val;
        }
        fftw_free(c);
        return SIS_OK;
    }

    /* Fast path: oversampled IDCT + cubic interpolation */
    int M = oversample * n;
    double* c = (double*)fftw_malloc(n * sizeof(double));
    if (!c) return SIS_ERR_NOMEM;
    int ret = sis_dct2(y, c, n);
    if (ret != SIS_OK) { fftw_free(c); return ret; }

    double inv_n = 1.0 / (double)n;
    double* pad = (double*)fftw_malloc(M * sizeof(double));
    if (!pad) { fftw_free(c); return SIS_ERR_NOMEM; }
    pad[0] = c[0] * 0.5 * inv_n * 0.5;
    for (int k = 1; k < n; ++k) pad[k] = c[k] * inv_n * 0.5;
    for (int k = n; k < M; ++k) pad[k] = 0.0;

    int idx = sis__ensure_plan(M);
    if (idx < 0) { fftw_free(pad); fftw_free(c); return SIS_ERR_NOMEM; }
    memcpy(sis__plans[idx].buf, pad, M * sizeof(double));
    fftw_execute_r2r(sis__plans[idx].idct2, sis__plans[idx].buf, sis__plans[idx].buf);
    double* fine = sis__plans[idx].buf; /* length M, unnormalised */

    double h = M_PI / (double)M;
    for (int i = 0; i < m; ++i) {
        double t = M_PI * sis__nufct_fold(xq[i], L) / L;
        double s = t / h - 0.5;
        int j = (int)floor(s);
        if (j < 0) j = 0;
        if (j >= M - 1) j = M - 2;
        double u = s - (double)j;
        out[i] = sis__nufct_interp_cubic(fine, M, j, u);
    }

    fftw_free(pad);
    fftw_free(c);
    return SIS_OK;
}

int sis_pepin1d_fit_denoised(sis_pepin1d_t* s, const double* x, const double* y,
                            int n, int degree, double cutoff_cycles) {
    double* yc = (double*)fftw_malloc(n * sizeof(double));
    if (!yc) return SIS_ERR_NOMEM;
    memcpy(yc, y, n * sizeof(double));
    int ret = sis_denoise_1d(yc, n, cutoff_cycles);
    if (ret == SIS_OK) ret = sis_pepin1d_build(s, x, yc, n, degree);
    fftw_free(yc);
    return ret;
}
#endif

/* ============================================================================
 * MIT License
 *
 * Copyright (c) 2026 Kevin Fling
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * ============================================================================ */
#endif /* SISYPHUS_IMPLEMENTATION */

#ifdef __cplusplus
}
#endif


