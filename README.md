# SISYPHUS

**Spectral Interpolation System Yielding Precise High-order Univariate Splines**

A single-dependency (FFTW3) header-only C11 / C++17 spectral toolbox for structured
and unstructured interpolation, exact spectral differentiation, high-degree
splines, denoising, and differentiable layers.

**Version:** v0.1.0  
**Repository:** <https://github.com/kevinfling/sisyphus.h>

```c
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
/* link with -lfftw3 -lm */
```

A thin C++17 convenience wrapper (`sisyphus.hpp`) provides RAII containers,
`std::vector` interop, and exception-based error handling:

```cpp
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.hpp"

std::vector<double> x = ... , y = ...;
sisyphus::Pepin1D spline(x, y, 7);
double v = spline.eval(0.5, 0);   // function value
double d = spline.eval(0.5, 1);   // first derivative
```

---

## Table of Contents

- [Overview](#overview)
- [Quick Example (C)](#quick-example-c)
- [Quick Example (C++)](#quick-example-c-1)
- [Mathematical Foundations](#mathematical-foundations)
  - [DCT-II and the Even Extension](#dct-ii-and-the-even-extension)
  - [Spectral Differentiation](#spectral-differentiation)
  - [Pepin High-Degree Splines](#pepin-high-degree-splines)
  - [Tensor-Product Splines](#tensor-product-splines)
  - [Thin-Plate Splines](#thin-plate-splines)
  - [Smoothing Splines via Demmler–Reinsch](#smoothing-splines-via-demmlerreinsch)
  - [Spectral Fractional Derivatives](#spectral-fractional-derivatives)
  - [Non-Uniform Fast Cosine Transform](#non-uniform-fast-cosine-transform)
  - [Barycentric Rational Interpolation](#barycentric-rational-interpolation)
  - [Weak Jacobians for Spline Layers](#weak-jacobians-for-spline-layers)
- [Building](#building)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Benchmarks](#benchmarks)
- [Tests, Benchmarks & Examples](#tests-benchmarks--examples)
- [License](#license)

---

## Overview

> There is but one truly serious numerical problem, and that is instability. All the rest — whether the grid is uniform or the noise is leptokurtic — comes later. The struggle toward high-order derivatives is enough to fill a man's heart. One must imagine the interpolant happy.

SISYPHUS provides a unified C11 API for spectral operations on structured and
unstructured data. Every algorithm is built on the discrete cosine transform of
type II (DCT-II) because its even extension eliminates Gibbs ringing at
non-periodic boundaries—something the DFT cannot do without artificial padding or
windowing.

Key features:

* **Exact round-trip DCT-II / IDCT-II** (FFTW3 REDFT10 / REDFT01) with proper
  normalization.
* **Separable ND DCT** for tensor-product grids in 1-D, 2-D and 3-D.
* **Spectral denoising** by hard low-pass filtering in the cosine basis.
* **Exact spectral differentiation** of the DCT interpolant to arbitrary integer
  order, and to arbitrary real order via fractional calculus.
* **Pepin high-degree splines** (parameter-free, $C^{\theta-1}$) built from spectral
  derivatives with $O(1)$ uniform-grid evaluation.
* **Tensor-product Pepin splines** on rectilinear grids (up to 3-D).
* **Thin-plate splines** (2-D and 3-D polyharmonic RBF) for scattered data.
* **Non-Uniform Fast Cosine Transform (NUFCT)** for $O(N \log N + M)$ evaluation
  of the DCT interpolant at arbitrary points.
* **Barycentric rational interpolation** (Floater–Hormann, pole-free) for
  irregular 1-D grids.
* **Smoothing splines** via spectral Demmler–Reinsch shrinkage.
* **Weak Jacobians** for piecewise-constant / piecewise-linear back-propagation.

---

## Quick Example (C)

```c
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    int n = 256;
    double x[256], y[256];
    for (int i = 0; i < n; ++i) {
        x[i] = (double)i / (n - 1);
        y[i] = sin(2.0 * M_PI * x[i]);
    }

    /* Build a degree-7 Pepin spline */
    sis_pepin1d_t s;
    sis_pepin1d_build(&s, x, y, n, 7);

    /* Evaluate and differentiate */
    double xq = 0.314;
    printf("f(%g)  = %g\n", xq, sis_pepin1d_eval(&s, xq, 0));
    printf("f'(%g) = %g\n", xq, sis_pepin1d_eval(&s, xq, 1));

    sis_pepin1d_free(&s);
    return 0;
}
```

Compile with:

```bash
clang -std=c11 -O3 -march=native -Iinclude example.c -lfftw3 -lm
```

## Quick Example (C++)

```cpp
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.hpp"
#include <cmath>
#include <vector>
#include <iostream>

int main() {
    int n = 256;
    double L = 1.0;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = L * i / (n - 1);
        y[i] = std::sin(2.0 * M_PI * x[i]);
    }

    sisyphus::Pepin1D spline(x, y, 7);

    double xq = 0.314;
    std::cout << "f(" << xq << ")  = " << spline.eval(xq, 0) << '\n';
    std::cout << "f'(" << xq << ") = " << spline.eval(xq, 1) << '\n';
    std::cout << "Integral = " << spline.integrate(0.0, L) << '\n';

    return 0;
}
```

Compile with:

```bash
clang++ -std=c++17 -O3 -march=native -Iinclude example.cpp -lfftw3 -lm
```

---

## Mathematical Foundations

### DCT-II and the Even Extension

Given samples $f_j = f(x_j)$ on the DCT-II nodes

$$
x_j = \frac{(j + \tfrac12)\,\pi}{N}, \qquad j = 0, \dots, N-1
$$

the DCT-II expansion is the unique *even* trigonometric interpolant

$$
\tilde{f}(t) = \frac{a_0}{2} + \sum_{k=1}^{N-1} a_k \cos(k t)
$$

with coefficients

$$
a_k = \frac{2}{N} \sum_{j=0}^{N-1} f_j \cos\!\left(\frac{\pi k (j+\tfrac12)}{N}\right), \qquad k = 0, 1, \dots, N-1 .
$$

The $k = 0$ term contributes $a_0/2 = \tfrac{1}{N}\sum_j f_j$ (the sample mean).

Because $\tilde{f}(t)$ is even about $t = 0$ and $t = \pi$, its periodic continuation is
$C^\infty$ except at the odd reflections, where all *odd* derivatives vanish.
Consequently there is **no Gibbs phenomenon** at the physical boundaries, making
the DCT-II the natural transform for non-periodic data.

FFTW3 implements the unnormalized DCT-II as `REDFT10`:

$$
Y_k = 2 \sum_{j=0}^{N-1} X_j \cos\!\left(\frac{\pi k (j+\tfrac12)}{N}\right) .
$$

Its inverse (DCT-III) is `REDFT01`:

$$
Z_j = Y_0 + 2 \sum_{k=1}^{N-1} Y_k \cos\!\left(\frac{\pi k (j+\tfrac12)}{N}\right) .
$$

These satisfy $Z = 2N X$.  The library normalizes `sis_idct2` so that
`sis_idct2(sis_dct2(X)) = X`, while `sis_dct2` returns the raw FFTW coefficients
(useful for library internals and power-spectrum analysis).

### Spectral Differentiation

The derivative of the cosine interpolant is obtained term-by-term in frequency
space:

$$
\frac{d^\mu}{dt^\mu}\!\left[\frac{a_0}{2} + \sum_{k=1}^{N-1} a_k \cos(k t)\right]
= \sum_{k=1}^{N-1} (-1)^{\lfloor (\mu+1)/2 \rfloor} \, k^\mu \, a_k \times
\begin{cases}
\cos(k t), & \mu \text{ even} \\
\sin(k t), & \mu \text{ odd}
\end{cases}
$$

Odd derivatives produce a *sine* series, which the library synthesises with
FFTW3's `RODFT01` (inverse DST-II).  The physical derivative on a domain
$[0, L]$ is recovered via the chain rule with scale factor $\sigma = \pi / L$:

$$
\frac{d^\mu}{dx^\mu} f(x) = \sigma^\mu \, \frac{d^\mu}{dt^\mu} \tilde{f}(t) .
$$

Because differentiation is diagonal in the cosine basis, it is **exact** (up to
machine precision) for the unique band-limited interpolant of the data.

### Pepin High-Degree Splines

A Pepin spline of degree $\theta$ on a partition $x_0 < x_1 < \dots < x_n$ is a piecewise
Taylor polynomial of degree $\theta$ on each interval $[x_j, x_{j+1})$:

$$
P_j(x) = \sum_{\mu=0}^{\theta} f^{(\mu)}(x_j) \, \frac{(x - x_j)^\mu}{\mu!} .
$$

By construction $P_j$ is $C^{\theta-1}$ at every knot (all derivatives up to order
$\theta-1$ match by definition).  The coefficients $f^{(\mu)}(x_j)$ are computed
**spectrally**: the library takes the sample vector $y_j = f(x_j)$, evaluates its
DCT-II, and applies the diagonal spectral differentiation matrix described above.
No linear system needs to be solved, no tension parameters are chosen, and the
method is stable for any $N$.

Evaluation uses Horner's rule on the shifted monomial basis.  For **uniform**
grids the interval location is $O(1)$ via direct index arithmetic; non-uniform
grids use binary search.  Analytic integration is performed by term-wise
antiderivation of the Taylor polynomial.

### Tensor-Product Splines

For a rectilinear grid $G = \{x_{0,i}\} \times \{x_{1,j}\} \times \dots \times \{x_{d-1,k}\}$ the mixed
partial derivatives

$$
\frac{\partial^{|\alpha|} f}{\partial x_0^{\alpha_0} \cdots \partial x_{d-1}^{\alpha_{d-1}}}, \qquad 0 \le \alpha_i \le \theta
$$

are computed by successive 1-D spectral differentiation along each axis.  The
$(\theta+1)^d$ partials are stored at every grid point.  Evaluation proceeds by
locating the containing hyper-rectangle and summing the appropriate Taylor
monomials.

### Thin-Plate Splines

For scattered centres $\xi_i \in \mathbb{R}^d$ ($d = 2$ or $3$) the thin-plate spline (TPS)
interpolant has the form

$$
s(x) = p(x) + \sum_{i=1}^{N} \beta_i \, \phi_d\bigl(\|x - \xi_i\|\bigr)
$$

where $p(x)$ is an affine polynomial and the radial basis function is

$$
\phi_2(r) = r^2 \log r \qquad (\text{2-D})
$$

$$
\phi_3(r) = r \qquad (\text{3-D polyharmonic})
$$

The coefficients $(\beta, p)$ satisfy the KKT system

$$
\begin{bmatrix}
K + \lambda I & P \\[4pt]
P^{\!\mathsf{T}} & 0
\end{bmatrix}
\begin{bmatrix}
\beta \\[4pt] c
\end{bmatrix}
=
\begin{bmatrix}
y \\[4pt] 0
\end{bmatrix}
$$

with $K_{ij} = \phi_d(\|\xi_i - \xi_j\|)$ and $P$ the polynomial Vandermonde.  The
library solves this dense system with LU factorisation and partial pivoting.  A
small $\lambda > 0$ regularises ill-conditioned centre configurations.

### Smoothing Splines via Demmler–Reinsch

A natural smoothing spline of order $2m-1$ minimises

$$
\sum_i \bigl|y_i - s(x_i)\bigr|^2 \;+\; \lambda \int_0^L \bigl|s^{(m)}(x)\bigr|^2 \, dx .
$$

In the DCT-II basis the penalty operator is diagonal because the cosine functions
are eigenfunctions of the iterated second-derivative operator with natural
(Neumann) boundary conditions.  The spectral filter is therefore

$$
\hat{s}_k = \frac{\bar{y}_k}{\,1 + \lambda \, (\sigma k)^{2m}}
$$

where $\bar{y}_k$ are the DCT-II coefficients of the data and $\sigma = \pi/L$.  This is the
Demmler–Reinsch spectral shrinkage estimator.  The implementation is $O(N \log N)$
and numerically stable for any $\lambda \ge 0$; $\lambda = 0$ reproduces the interpolant.

### Spectral Fractional Derivatives

For a real order $\alpha > 0$ the fractional derivative of the cosine interpolant is
obtained via the Riemann–Liouville/Fourier symbol:

$$
D^\alpha \cos(k t) = k^\alpha \cos\!\left(\frac{\alpha\pi}{2}\right) \cos(k t) \;-\; k^\alpha \sin\!\left(\frac{\alpha\pi}{2}\right) \sin(k t)
$$

$$
D^\alpha \sin(k t) = k^\alpha \sin\!\left(\frac{\alpha\pi}{2}\right) \cos(k t) \;+\; k^\alpha \cos\!\left(\frac{\alpha\pi}{2}\right) \sin(k t)
$$

The even part of the data is transformed with DCT-II → IDCT-II and weighted by
$k^\alpha \cos(\alpha\pi/2)$; the odd part is transformed with DCT-II → IDST-II and weighted
by $k^\alpha \sin(\alpha\pi/2)$.  The two contributions are superposed with the $\tfrac12$ factor
inherent to REDFT01/RODFT01.  When $\alpha$ is within $10^{-6}$ of an integer the
function automatically snaps to the exact integer-derivative path (no
trigonometric round-off).

### Non-Uniform Fast Cosine Transform

Evaluating the DCT-II interpolant at $M$ arbitrary points by direct cosine
summation costs $O(NM)$.  The NUFCT reduces this to $O(N \log N + M)$ by:

1. Zero-padding the DCT coefficients to an oversampled grid of length $M_{\text{ov}} =
   \text{oversample} \cdot N$.
2. Applying an IDCT-II on the fine grid (one FFTW plan).
3. Locating each query point in the fine grid and reading the value through
   local cubic Lagrange interpolation.

For small problems ($N \cdot M < 10\,000$) the routine transparently falls back
to direct cosine summation.

### Barycentric Rational Interpolation

For irregular 1-D grids where spectral differentiation is not appropriate, the
Floater–Hormann barycentric rational interpolator provides a pole-free,
Lebesgue-constant-bounded alternative.  Given knots $x_0 < \dots < x_n$ and
blending degree $d$, the library uses the equispaced-form weights

$$
w_j = (-1)^{j-d} \sum_{i = \max(0,\,j-d)}^{\min(j,\,n-d)} \binom{d}{j - i}
$$

and evaluation proceeds in the barycentric form

$$
P(x) = \frac{\displaystyle\sum_j \frac{w_j \, f_j}{x - x_j}}
             {\displaystyle\sum_j \frac{w_j}{x - x_j}} .
$$

The binomial coefficients are accumulated in double precision to avoid integer
overflow for $d \ge 5$.

### Weak Jacobians for Spline Layers

Differentiable spline layers require back-propagating gradients through the
spline basis.  For piecewise-constant (PWC) and piecewise-linear (PWL) bases the
Jacobian is block-diagonal.  The library provides the weak forms derived by
Cho et al.:

* **PWC**: each sample $i$ inherits the upstream gradient of its block,
  $\text{grad\_out}[i] = \text{grad\_in}[\text{partition}[i]]$.
* **PWL**: hat-function blend of the two endpoint gradients on each interval,
  $\text{grad\_out}[i] = a \cdot \text{grad\_in}[j] + b \cdot \text{grad\_in}[j+1]$,
  where $j$ is the interval containing $x_i$ and $a = (t_{j+1} - x_i)/h$,
  $b = (x_i - t_j)/h$, $h = t_{j+1} - t_j$.

---

## Building

```bash
cmake -B build -S .
cmake --build build -j
```

Requirements:
* C11 compiler (Clang ≥ 14, GCC ≥ 9) or C++17 compiler (Clang ≥ 14, GCC ≥ 9)
* FFTW3 development libraries (`libfftw3-dev` on Debian/Ubuntu)
* CMake ≥ 3.16

CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTING` | `ON` | Build the test suite |
| `BUILD_BENCHMARKS` | `ON` | Build performance benchmarks |
| `BUILD_EXAMPLES` | `ON` | Build example programs |

Installation:

```bash
cmake --install build --prefix /path/to/install
```

Because the library is header-only, the install step copies `sisyphus.h` and
`sisyphus.hpp` to `${prefix}/include` and exports a CMake target `sisyphus::spectral`.

### Single-file usage

If you prefer not to use CMake, include the header and define the implementation
macro in exactly one translation unit:

**C:**
```c
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
/* link with -lfftw3 -lm */
```

**C++:**
```cpp
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.hpp"
/* link with -lfftw3 -lm */
```

---

## Usage Examples

### Spectral denoising (C)

```c
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"

/* noisy[0..n-1] is your contaminated signal */
sis_denoise_1d(noisy, n, 0.05);   /* keep lowest 5 % of modes */
```

### Spectral denoising (C++)

```cpp
std::vector<double> noisy = ...;
sisyphus::denoise(noisy, 0.05);
```

### Fractional derivative of a Gaussian (C)

```c
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"

/* f[0..n-1] sampled on [0, L]; out[0..n-1] receives D^alpha f */
double alpha = 1.5;
double scale = M_PI / L;
sis_fractional_derivative_1d(f, out, n, alpha, scale);
```

### Fractional derivative of a Gaussian (C++)

```cpp
std::vector<double> f = /* ... */;
std::vector<double> dalpha;
sisyphus::fractional_derivative(f, dalpha, 1.5, L);
```

### 2-D tensor-product Pepin spline (C)

```c
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"

int nx = 32, ny = 32;
int n[2] = {nx, ny};
double xk[2][32];          /* grid coordinates */
double y[32 * 32];         /* row-major samples */
/* ... fill xk[0], xk[1] and y ... */

sis_tensor_t s;
sis_tensor_init(&s, 2, n, 7);              /* allocate for degree-7, 2-D */
const double* xp[2] = {xk[0], xk[1]};
sis_tensor_build(&s, xp, y, 7);

double xq[2] = {0.5, 0.5};
double val = sis_tensor_eval(&s, xq, -1);  /* function value */
double dx  = sis_tensor_eval(&s, xq, 0);   /* ∂/∂x */

sis_tensor_free(&s);
```

### 2-D tensor-product Pepin spline (C++)

```cpp
std::vector<std::vector<double>> coords = { x_grid, y_grid };
std::vector<double> y = ...;   /* row-major, size nx*ny */

sisyphus::Tensor spline(coords, y, 7);
std::vector<double> xq = {0.5, 0.5};
double val = spline.eval(xq, -1);   /* function value */
double dx  = spline.eval(xq, 0);    /* ∂/∂x */
```

### Thin-plate spline from scattered centres (C)

```c
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"

/* xi: 2×N array of centres, yi: N values */
sis_tps_t tps;
sis_tps_fit(&tps, xi, yi, n, 2, 1e-6);   /* 2-D, small regularisation */

double xq[2] = {0.3, 0.7};
double val = sis_tps_eval(&tps, xq);

sis_tps_free(&tps);
```

### Thin-plate spline from scattered centres (C++)

```cpp
std::vector<double> xi = ...;   /* row-major, shape [n][d] */
std::vector<double> yi = ...;

sisyphus::ThinPlateSpline tps(xi, yi, 2, 1e-6);
std::vector<double> xq = {0.3, 0.7};
double val = tps.eval(xq);
```

### NUFCT: evaluate DCT interpolant at non-uniform points (C)

```c
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"

/* y[0..n-1] on [0, L], m query points xq[0..m-1] */
sis_nufct_eval(n, y, m, xq, out, L, 16);   /* 16× oversampling */
```

### NUFCT: evaluate DCT interpolant at non-uniform points (C++)

```cpp
std::vector<double> y = ... , xq = ...;
auto out = sisyphus::nufct_eval(y, xq, L, 16);
```

---

## API Reference

### C API

All functions return an `int` status code (`SIS_OK = 0` on success).

#### DCT

```c
int sis_dct2 (const double* in, double* out, int n);
int sis_idct2(const double* in, double* out, int n);

int sis_dct2_nd (int ndim, const int* dims, const double* in,
                double* out, double* work);
int sis_idct2_nd(int ndim, const int* dims, const double* in,
                double* out, double* work);
```

`ndim` may be 1, 2 or 3.  `work` may be `NULL`.

#### Denoising

```c
int sis_denoise_1d(double* inout, int n, double cutoff_cycles);
int sis_denoise_nd(int ndim, const int* dims, double* inout,
                  double cutoff_cycles, double* work);
```

`cutoff_cycles` is the fraction of modes to retain (e.g. `0.05` keeps the lowest
5 %).

#### Spectral Differentiation

```c
int sis_derivative_1d(const double* in, double* out, int n,
                     int mu, double scale);
int sis_derivative_nd(int ndim, const int* dims, const double* in,
                     double* out, int dir, int mu, double scale,
                     double* work);
```

`scale = π / L` for a physical domain $[0, L]$.  `dir` is the axis index
(0-based).

#### Fractional Derivative

```c
int sis_fractional_derivative_1d(const double* in, double* out, int n,
                                double alpha, double scale);
```

`alpha` may be any non-negative real number.  For integer values the function
automatically delegates to the exact integer-derivative path.

#### NUFCT

```c
int sis_nufct_eval(int n, const double* y, int m, const double* xq,
                  double* out, double L, int oversample);
```

Evaluate the DCT-II interpolant of `y[0..n-1]` at `m` non-uniform query points
`xq` on the physical domain $[0, L]$.  `oversample` is the grid refinement factor
(e.g. `8` or `16`); use `0` or negative for automatic selection.  For small
problems the routine falls back to direct cosine summation.

#### Barycentric Rational Interpolation

```c
typedef struct {
    int n;          /* knots - 1 */
    int d;          /* blending degree */
    double* x;      /* knots */
    double* f;      /* values */
    double* w;      /* weights */
} sis_barycentric_t;

int    sis_barycentric_init (sis_barycentric_t* s, int npoints, int degree);
void   sis_barycentric_free (sis_barycentric_t* s);
int    sis_barycentric_build(sis_barycentric_t* s, const double* x,
                            const double* y, int n, int degree);
double sis_barycentric_eval (const sis_barycentric_t* s, double xq);
int    sis_barycentric_fit_denoised(sis_barycentric_t* s, const double* x,
                                   const double* y, int n, int degree,
                                   double cutoff_cycles);
```

#### Pepin Spline (1-D)

```c
typedef struct {
    int n;          /* intervals */
    int degree;     /* θ */
    int uniform;    /* non-zero if grid is uniform (O(1) lookup) */
    double* x;      /* knots, size n+1 */
    double* c;      /* Taylor coeffs [interval][θ+1] */
} sis_pepin1d_t;

int  sis_pepin1d_init (sis_pepin1d_t* s, int npoints, int degree);
void sis_pepin1d_free (sis_pepin1d_t* s);
int  sis_pepin1d_build(sis_pepin1d_t* s, const double* x,
                      const double* y, int n, int degree);
double sis_pepin1d_eval(const sis_pepin1d_t* s, double xq, int deriv);
double sis_pepin1d_integrate(const sis_pepin1d_t* s, double a, double b);

/* Convenience: denoise then fit */
int sis_pepin1d_fit_denoised(sis_pepin1d_t* s, const double* x,
                            const double* y, int n, int degree,
                            double cutoff_cycles);
```

#### Tensor-Product Spline

```c
typedef struct {
    int ndim;
    int degree;
    int n[SIS_MAX_DIM];
    double* xk[SIS_MAX_DIM];
    double* mixed;
    size_t mixed_stride;
} sis_tensor_t;

int    sis_tensor_init (sis_tensor_t* s, int ndim, const int* n, int degree);
void   sis_tensor_free (sis_tensor_t* s);
int    sis_tensor_build(sis_tensor_t* s, const double* xk[SIS_MAX_DIM],
                       const double* y, int degree);
double sis_tensor_eval (const sis_tensor_t* s, const double* xq, int deriv_dir);
```

`deriv_dir = -1` evaluates the function; `0, 1, 2` evaluates the first
derivative along that axis.  Supported `ndim`: 1, 2, or 3.

#### Thin-Plate Spline

```c
typedef struct { int n, d; double* xi; double* beta; double* poly; } sis_tps_t;

int    sis_tps_init(sis_tps_t* s, int n, int d);
void   sis_tps_free(sis_tps_t* s);
int    sis_tps_fit (sis_tps_t* s, const double* xi, const double* yi,
                   int n, int d, double lambda);
double sis_tps_eval(const sis_tps_t* s, const double* xq);
```

#### Smoothing Spline

```c
int sis_smoothing_spline_1d(const double* x, const double* y, int n,
                           int order, double lambda, double* out_smooth);
```

`order = 3` gives a cubic smoothing spline ($m = 2$).

#### Weak Jacobians

```c
int sis_weak_jacobian_pwc(int n, const int* partition, int k,
                         const double* grad_in, double* grad_out);
int sis_weak_jacobian_pwl(int n, const double* x, const double* t, int k,
                         const double* grad_in, double* grad_out);
```

#### Lifecycle

```c
void sis_fftw_init(void);      /* optional; called automatically */
void sis_fftw_cleanup(void);   /* free all cached FFTW plans */
```

### C++ API

The C++ header `sisyphus.hpp` wraps the C API with RAII, `std::vector`, and
exceptions (`sisyphus::Exception`).

**Stateless free functions:**

```cpp
namespace sisyphus {
    void dct2 (const std::vector<double>& in, std::vector<double>& out);
    void idct2(const std::vector<double>& in, std::vector<double>& out);
    void dct2_nd (const std::vector<int>& dims,
                  const std::vector<double>& in, std::vector<double>& out);
    void idct2_nd(const std::vector<int>& dims,
                  const std::vector<double>& in, std::vector<double>& out);

    void denoise   (std::vector<double>& inout, double cutoff_cycles);
    void denoise_nd(const std::vector<int>& dims,
                    std::vector<double>& inout, double cutoff_cycles);

    void derivative   (const std::vector<double>& in, std::vector<double>& out,
                       int mu, double domain_length);
    void derivative_nd(const std::vector<int>& dims,
                       const std::vector<double>& in, std::vector<double>& out,
                       int dir, int mu, double domain_length);
    void fractional_derivative(const std::vector<double>& in,
                               std::vector<double>& out,
                               double alpha, double domain_length);

    std::vector<double> smoothing_spline(const std::vector<double>& x,
                                         const std::vector<double>& y,
                                         int order, double lambda);
    std::vector<double> nufct_eval(const std::vector<double>& y,
                                   const std::vector<double>& xq,
                                   double L, int oversample = 0);

    std::vector<double> weak_jacobian_pwc(const std::vector<int>& partition,
                                          int k,
                                          const std::vector<double>& grad_in);
    std::vector<double> weak_jacobian_pwl(const std::vector<double>& x,
                                          const std::vector<double>& t,
                                          int k,
                                          const std::vector<double>& grad_in);
}
```

**RAII wrappers:**

| Class | Key methods |
|-------|-------------|
| `Pepin1D` | `eval(xq, deriv)`, `integrate(a, b)`, `fit_denoised(x, y, degree, cutoff)` |
| `Tensor` | `eval(xq, deriv_dir)` — `std::vector` and `std::array` overloads |
| `ThinPlateSpline` | `eval(xq)` — `std::vector` and `std::array` overloads |
| `Barycentric` | `eval(xq)`, `fit_denoised(x, y, degree, cutoff)` |

All wrappers are **move-only** (deleted copy, movable), and clean up C resources
in their destructors.

---

## Benchmarks

All numbers were measured on a single core of an AMD Ryzen 9 7950X with Clang
22.1.3 (`-O3 -march=native`).

### DCT-II round-trip throughput

| N    | Time (μs) | Throughput (Melem/s) |
|------|-----------|----------------------|
| 64   | 1.08      | 59.4                 |
| 256  | 4.67      | 54.8                 |
| 1024 | 45.1      | 22.7                 |
| 4096 | 168.6     | 24.3                 |

The drop at N = 1024 reflects FFTW plan creation overhead on the first call;
subsequent calls with the same length reuse cached plans.

### Pepin spline evaluation (degree 7, uniform grid)

| N    | Time / query |
|------|--------------|
| 256  | 33 ns        |
| 1024 | 33 ns        |
| 4096 | 41 ns        |

Evaluation time is essentially **flat** with N because interval location on a
uniform grid is $O(1)$.  The small increase at N = 4096 is due to the larger
Taylor-coefficient working set.

### Pepin convergence vs. grid type

Fitting $\sin(2\pi x)$ on $[0, 1]$ with a degree-11 Pepin spline:

| N    | Grid type | L₁ error    | Max error   | Time / query |
|------|-----------|-------------|-------------|--------------|
| 256  | Uniform   | 3.93×10⁻³   | 1.33×10⁻²   | 58 ns        |
| 256  | DCT-II    | 1.08×10⁻⁴   | 9.05×10⁻³   | 56 ns        |
| 1024 | Uniform   | 9.79×10⁻⁴   | 3.34×10⁻³   | 61 ns        |
| 1024 | DCT-II    | 6.24×10⁻⁶   | 2.76×10⁻⁴   | 59 ns        |
| 4096 | Uniform   | 2.45×10⁻⁴   | 7.65×10⁻⁴   | 71 ns        |
| 4096 | DCT-II    | 3.95×10⁻⁷   | 5.48×10⁻⁵   | 69 ns        |

On a **uniform grid** the error decreases as $O(1/N)$ because spectral
differentiation is not exact at uniform samples.  On the **DCT-II grid**
$x_j = (j+\tfrac12)\pi/N$ the error is orders of magnitude smaller (up to 600× at
N = 4096) because the DCT-II interpolant and its derivatives are evaluated at
their natural collocation nodes.

### NUFCT (n = 1024, m = 100 000)

| Method            | Time   | Time / query |
|-------------------|--------|--------------|
| Fast (16× oversample) | 6.57 ms | 65.7 ns |
| Direct cosine sum     | 16.2 ms | 162 ns  |

### Spectral fractional derivative (n = 4096)

| α    | Time (μs) |
|------|-----------|
| 0.5  | 521       |
| 1.0  | 159       |
| 1.5  | 543       |
| 2.0  | 181       |

Integer orders are faster because they require only one transform pair; fractional
orders need two (even + odd) plus a final combination pass.

### Other modules

| Module                | Configuration         | Time          |
|-----------------------|-----------------------|---------------|
| Barycentric rational  | d = 5, n = 256        | 1351 ns/query |
| Tensor-product Pepin  | deg 3, 32×32 grid     | 1.35 ns/query |
| TPS eval              | n = 100 centres, 2-D  | 1923 ns/query |
| TPS fit               | n = 100 centres, 2-D  | 521 μs total  |

---

## Tests, Benchmarks & Examples

### Tests

```bash
cd build
ctest --output-on-failure
```

Eleven C test suites and one C++ wrapper test suite (`test_cpp_wrapper`)
cover DCT roundtrips, denoising, spectral differentiation, fractional derivatives,
Pepin splines, tensor-product splines, TPS fitting, smoothing splines, weak
Jacobians, barycentric rational interpolation, and NUFCT accuracy.

### Benchmarks

| Binary | What it measures |
|--------|------------------|
| `bench_dct` | 1-D DCT-II + IDCT-II throughput vs. grid size |
| `bench_pepin` | Pepin spline evaluation throughput (degree 7) |
| `bench_tensor` | Tensor-product build + eval (2-D, degree 3) |
| `bench_tps` | TPS fit + eval (100 centres, 2-D) |
| `bench_fractional` | Spectral fractional derivative for several α |
| `bench_barycentric` | Barycentric rational eval throughput (Runge) |
| `bench_nufct` | Fast NUFCT vs. direct cosine summation |
| `bench_cpp_wrapper` | C++ wrapper overhead (Pepin, Tensor, TPS) |
| `bench_pepin_convergence` | Accuracy and speed vs. N and degree |

Run individually from `build/benchmarks/`.

### Examples

| Binary | Description |
|--------|-------------|
| `example_denoise_1d` | Hard low-pass filter on a noisy sine wave |
| `example_denoise_2d` | Separable DCT denoising on a 64×64 grid |
| `example_spline_fit` | Fit a degree-7 Pepin spline and print derivatives |
| `example_scattered` | TPS interpolation of 50 random centres in 2-D |
| `example_fractional` | Fractional derivatives of a Gaussian bump |
| `example_barycentric` | Rational vs. Pepin interpolation of the Runge function |
| `example_nufct` | Evaluate a denoised signal at $10^4$ random points |
| `example_cpp_spline_fit` | C++17 Pepin spline fit, derivatives, integral |

---

## License

MIT License

Copyright (c) 2026 Kevin Fling

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
