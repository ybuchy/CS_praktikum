#define SIZE 1000000
#define TYPE double

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

template<typename T, uint32_t n>
void read_data2(T* A, T* x) {
    std::ifstream file_A("../data/A.data");
    for (uint32_t i = 0; i < n * n; i += 1) {
        file_A >> A[i];
    }
    std::ifstream file_x("../data/x.data");
    for (uint32_t i = 0; i < n; i += 1) {
        file_x >> x[i];
    }
}

template<typename T, uint32_t n>
void read_data(T* vec, int data_num) {
}

// & for pass by reference, otherwise whole thing gets copied
template<typename T>
void read_data(std::vector<std::vector<T>>& vectors) {
    std::cout << "current n: " << SIZE << std::endl;

    std::cout << "Waiting for data generation..." << std::endl;
    std::cin.get();

    for (uint32_t i = 0; i < vectors.size(); i += 1) {
        std::ifstream file_vec(std::format("../data/data{}.data", i));
        for (uint32_t j = 0; j < vectors[i].size(); j += 1) {
            file_vec >> vectors[i][j];
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

// [? TODO] write this with function given as parameter? probably can't w/ templates
void test_addVec() {
    std::cout << "current n: " << SIZE << std::endl;

    std::cout << "Waiting for data generation..." << std::endl;
    std::cin.get();

    std::vector<double> u(SIZE), v(SIZE), w(SIZE);

    std::ifstream file_v("../data/vec1.data");
    for (uint32_t i = 0; i < SIZE; i += 1) {
        file_v >> v[i];
    }

    std::ifstream file_w("../data/vec2.data");
    for (uint32_t i = 0; i < SIZE; i += 1) {
        file_w >> w[i];
    }

    auto start = std::chrono::high_resolution_clock::now();
    prim_omp::addVec<double, SIZE>(u.data(), v.data(), w.data());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    prim::addVec<double, SIZE>(u.data(), v.data(), w.data());
    stop = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "openmp one: " << duration1.count() << " normal one: " << duration2.count() << std::endl;
}

void test_facVec() {
    double u[SIZE] = {};
    double v[SIZE] = {};
    const double c = 9129412.81281;

    // [? TODO] random vectors
    for(int i = 0; i < SIZE; i++) v[i] = i + 1;
    auto start = std::chrono::high_resolution_clock::now();
    prim::facVec<double, SIZE>(u, c, v);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    prim_omp::facVec<double, SIZE>(u, c, v);
    stop = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "normal one: " << duration1.count() << " openmp one: " << duration2.count() << std::endl;
}

void test_cg() {
    std::cout << "current n: " << SIZE << std::endl;
    
    std::cout << "Waiting for data generation..." << std::endl;
    std::cin.get();

    double A[SIZE];//*SIZE];
    double x[SIZE];
    //read_data<double, SIZE>(A, x);

    double b[SIZE];
    prim_omp::matVec<double, SIZE>(b, A, x);
    double sol[SIZE] = {};

    auto start = std::chrono::high_resolution_clock::now();
    int k = cg<double, SIZE>(sol, A, b);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "return value: " << k << std::endl;
    std::cout << "cg solution:" << std::endl;
    print_vector<double, SIZE>(sol);
    std::cout  << "real solution:" << std::endl;
    print_vector<double, SIZE>(x);
    std::cout << "duration: " << duration.count() << std::endl;
}

int main(int argc, char* argv[]) {
    std::string arg = argv[1];
    if (arg != "--function") return 1;
    std::string fn = argv[2];
    if (fn == "addVec") {
        std::vector<TYPE> u(SIZE), v(SIZE), w(SIZE);
        std::vector<std::vector<TYPE>> vectors{v, w};
        read_data(vectors);
        auto f1 = &prim::addVec<TYPE, SIZE>;
        auto f2 = &prim_omp::addVec<TYPE, SIZE>;
        compare(f1, f2, u.data(), v.data(), w.data(), false);
    } else if (fn == "facVec") {
        std::vector<TYPE> u(SIZE), v(SIZE);
        std::vector<std::vector<TYPE>> vectors{v};
        read_data(vectors);
        TYPE c = std::numbers::pi;
        auto f1 = &prim::facVec<TYPE, SIZE>;
        auto f2 = &prim_omp::facVec<TYPE, SIZE>;
        compare(f1, f2, u.data(), c, v.data());
    } else if (fn == "dot") {

    } else if (fn == "matVec") {

    } else if (fn == "cg") {

    }
}
