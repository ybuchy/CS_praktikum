#pragma once
#include <cmath>
#include <cstdint>
#include <primitives_OpenMP.h>


namespace prim_omp {
    // initialized (n x n) identity matrix at given array pointer
    template<typename T, uint32_t n>
    void identity(T* M) {
        for (uint32_t i = 0; i < n * n; i += 1) {
            M[i] = 0;
        }
        for (uint32_t i = 0; i < n; i += 1) {
            M[i * n + i] = 1;
        }
    }

    // [? TODO] is uint32_t too small for maxiter?, also n*10 from numpy too small for standard value?
    // Returns 0 if close enough to solution, maxiter if didn't find sol before exhausting number of iterations (see numpy cg)
    template<typename T, uint32_t n>
    int cg(T* x, const T* A, const T* b, T* x_0=nullptr,  const double rtol=1e-5, const double atol=0, uint32_t maxiter=10*n, T* M=nullptr) {

        if (x_0 == nullptr)  x_0 = new T[n]();
        if (M == nullptr) {
            M = new T[n*n]();
            identity<T, n>(M);
        }

        T b_norm = sqrt(prim_omp::dot<T, n>(b, b));
        T tol = std::max(atol, rtol * b_norm);

        T Ax_0[n];
        prim_omp::matVec<T, n>(Ax_0, A, x_0);
        T r[n];
        prim_omp::addVec<T, n>(r, b, Ax_0, true);



        // initialize needed variables
        T rho_prev = static_cast<T>(0);
        T rho_cur = static_cast<T>(0);
        T r_norm = static_cast<T>(0);
        T alpha = static_cast<T>(0);
        T p[n] = {};
        T q[n] = {};
        T ap[n] = {};
        T aq[n] = {};
        T z[n] = {};
    
        for (uint32_t iter = 0; iter < maxiter; iter += 1) {
            r_norm = sqrt(prim_omp::dot<T, n>(r, r));
            if (r_norm < tol) {
                if (iter == 0) {
                    for (uint32_t i = 0; i < n; i += 1) x[i] = x_0[i];
                }
                return 0;
            }
            prim_omp::matVec<T, n>(z, M, r);
            rho_cur = prim_omp::dot<T, n>(r, z);
            if (iter > 0) {
                prim_omp::facVec<T, n>(p, rho_cur / rho_prev, p);
                prim_omp::addVec<T, n>(p, p, z);
            } else {
                for(uint32_t i = 0; i < n; i += 1) {
                    p[i] = z[i];
                }
            }
            prim_omp::matVec<T, n>(q, A, p);
            alpha = rho_cur / prim_omp::dot<T, n>(p, q);
            prim_omp::facVec<T, n>(ap, alpha, p);
            prim_omp::addVec<T, n>(x, x, ap);
            prim_omp::facVec<T, n>(aq, alpha, q);
            prim_omp::addVec<T, n>(r, r, aq, true);
            rho_prev = rho_cur;
        }
        return maxiter;
    }
}
