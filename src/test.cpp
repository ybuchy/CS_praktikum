#define TYPE double

#include <cstddef>
#include <iostream>
#include <numbers>
#include <functional>
#include <format>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <primitives.h>
#include <primitives_OpenMP.h>
#include <solvers.h>
#include <solvers_OpenMP.h>

constexpr size_t SIZE = 100;

// Output the matrix A which is (n x m)
template<typename T, uint32_t n, uint32_t m>
void print_matrix(T* A) {
    // rows
    for (uint32_t i = 0; i < n; i += 1) {
        // cols
        for (uint32_t j = 0; j < m; j += 1) {
            std::cout << A[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Output vector v of length n
template<typename T, uint32_t n>
void print_vector(T* v) {
    for (uint32_t i = 0; i < n; i += 1) {
        std::cout << v[i] << std::endl;
    }
}

// & for pass by reference, otherwise whole thing gets copied
template<typename T>
void read_data(std::vector<std::vector<T>>& vectors, std::vector<std::vector<T>>& matrices) {
    std::cout << "current n: " << SIZE << std::endl;
    std::cout << "num vectors: " << vectors.size() << std::endl;
    std::cout << "num matrices: " << matrices.size() << std::endl;
    std::cout << "Waiting for data generation..." << std::endl;
    std::cin.get();

    for (uint32_t i = 0; i < vectors.size(); i += 1) {
        std::ifstream file_vec(std::format("../data/vec{}.data", i));
        for (uint32_t j = 0; j < vectors[i].size(); j += 1) {
            file_vec >> vectors[i][j];
        }
    }

    for (uint32_t i = 0; i < matrices.size(); i += 1) {
        std::ifstream file_mat(std::format("../data/mat{}.data", i));
        for (uint32_t j = 0; j < matrices[i].size(); j += 1) {
            file_mat >> matrices[i][j];
        }
    }
}

template<typename F1, typename F2, typename... Args>
void compare(F1&& f1, F2&& f2, Args&&... args) {
    // is this forward thing a problem for checking time?
    auto start = std::chrono::high_resolution_clock::now();
    f1(std::forward<Args>(args)...);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    f2(std::forward<Args>(args)...);
    stop = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "first one: " << duration1.count() << " second one: " << duration2.count() << std::endl;
    
}

int main(int argc, char* argv[]) {
    std::string arg = argv[1];
    if (arg != "--function") return 1;
    std::string fn = argv[2];
    if (fn == "addVec") {
        std::vector<TYPE> u(SIZE), v(SIZE), w(SIZE);
        std::vector<std::vector<TYPE>> vectors{v, w};
        std::vector<std::vector<TYPE>> matrices{};
        read_data(vectors, matrices);
        auto f1 = &prim::addVec<TYPE, SIZE>;
        auto f2 = &prim_omp::addVec<TYPE, SIZE>;
        compare(f1, f2, u.data(), v.data(), w.data(), false);
    } else if (fn == "facVec") {
        std::vector<TYPE> u(SIZE), v(SIZE);
        std::vector<std::vector<TYPE>> vectors{v};
        std::vector<std::vector<TYPE>> matrices{};
        read_data(vectors, matrices);
        TYPE c = std::numbers::pi;
        auto f1 = &prim::facVec<TYPE, SIZE>;
        auto f2 = &prim_omp::facVec<TYPE, SIZE>;
        compare(f1, f2, u.data(), c, v.data());
    } else if (fn == "dot") {
        std::vector<TYPE> v(SIZE), w(SIZE);
        std::vector<std::vector<TYPE>> vectors{v, w};
        std::vector<std::vector<TYPE>> matrices{};
        read_data(vectors, matrices);
        auto f1 = &prim::dot<TYPE, SIZE>;
        auto f2 = &prim_omp::dot<TYPE, SIZE>;
        compare(f1, f2, v.data(), w.data());
    } else if (fn == "matVec") {
        std::vector<TYPE> w(SIZE), A(SIZE * SIZE), v(SIZE);
        std::vector<std::vector<TYPE>> vectors{v};
        std::vector<std::vector<TYPE>> matrices{A};
        read_data(vectors, matrices);
        auto f1 = &prim::matVec<TYPE, SIZE>;
        auto f2 = &prim_omp::matVec<TYPE, SIZE>;
        compare(f1, f2, w.data(), A.data(), v.data());
    } else if (fn == "cg") {
        std::vector<TYPE> x(SIZE), A(SIZE * SIZE), b(SIZE);
        std::vector<std::vector<TYPE>> vectors{x, b};
        std::vector<std::vector<TYPE>> matrices{A};
        read_data(vectors, matrices);
        auto f1 = &prim::cg<TYPE, SIZE>;
        auto f2 = &prim_omp::cg<TYPE, SIZE>;
        compare(f1, f2, x.data(), A.data(), b.data(), nullptr, 1e-5, 0, 10*SIZE, nullptr);
    }
}
