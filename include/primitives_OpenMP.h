#pragma once
#include <cstdint>

namespace prim_omp {
    // Compute the matrix vector product w = Av where A is (n x n), v has length n
    // [? TODO] also do non-quadratic matrices?
    template<typename T, uint32_t n> 
    void matVec(T* w, const T* A, const T* v) {
        // [? TODO] Why declare here?
        T sum;
#pragma omp parallel for
        for (uint32_t i = 0; i < n; i += 1) {
            // Matrix is stored in contiguous memory block
            sum = static_cast<T>(0);
            for (uint32_t j = 0; j < n; j++) {
                sum += A[i * n + j] * v[j];
            }
            w[i] = sum;
        }
    }

    // Compute the dot product u^T * v
    template<typename T, uint32_t n>
    T dot(const T* u, const T* v) {
        T sum = static_cast<T>(0);
        // [? TODO] sum thing!
#pragma omp parallel for reduction(+:sum)
            for (uint32_t j = 0; j < n; j += 1) {
                sum += u[j] * v[j];
            }
        return sum;
    }


    // Compute u = c * v elementwise multiplication with c scalar of type T, v vector of length n
    template<typename T, uint32_t n>
    void facVec(T* u, const T c, const T* v) {
#pragma omp parallel for
        for (uint32_t i = 0; i < n; i += 1) {
            u[i] = c * v[i];
        }
    }

    // Compute w = u + v for vectors of length n, if negative = true compute u - v
    template<typename T, uint32_t n>
    void addVec(T* w, const T* u, const T* v, bool negative=false) {
        T factor;
        if (negative == true) {
            factor = static_cast<T>(-1);
        } else {
            factor = static_cast<T>(1);
        }
#pragma omp parallel for
            for (uint32_t i = 0; i < n; i += 1) {
                w[i] = u[i] + factor * v[i];
            }
    }
}
