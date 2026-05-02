#define SIZE 1000

#include <iostream>
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
void read_data(T* A, T* x) {
   std::ifstream file_A("../data/A.data");
    for (uint32_t i = 0; i < n * n; i += 1) {
        file_A >> A[i];
    }
    std::ifstream file_x("../data/x.data");
    for (uint32_t i = 0; i < n; i += 1) {
        file_x >> x[i];
    }
}

// [? TODO] write this with function given as parameter? probably can't w/ templates
void test_addVec() {
    double u[SIZE] = {};
    double v[SIZE] = {};
    double w[SIZE] = {};

    // [? TODO] random vectors
    for(int i = 0; i < SIZE; i++) v[i] = i + 1;
    for(int i = 0; i < SIZE; i++) w[i] = SIZE - i;
    auto start = std::chrono::high_resolution_clock::now();
    prim::addVec<double, SIZE>(u, v, w);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    prim_omp::addVec<double, SIZE>(u, v, w);
    stop = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << duration1.count() << " " << duration2.count() << std::endl;
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

    std::cout << duration1.count() << " " << duration2.count() << std::endl;
}

void test_cg() {
    std::cout << "current n: " << SIZE << std::endl;
    
    std::cout << "Waiting for data generation..." << std::endl;
    std::cin.get();

    double A[SIZE*SIZE];
    double x[SIZE];
    read_data<double, SIZE>(A, x);

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
    // [? TODO] do switch case to say what u wanna test
    test_addVec();
}
