#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>
#include <immintrin.h>

using std::vector;

const double tol = 1e-8; // tolerance
const int max_iter = 100; // maximum number of iterations

void chol_factorization(vector<vector<double>>& A) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++) {
                A[i][j] -= A[i][k] * A[j][k];
            }
            A[i][j] /= A[j][j];
        }
        for (int k = 0; k < i; k++) {
            A[i][i] -= A[i][k] * A[i][k];
        }
        A[i][i] = sqrt(A[i][i]);
    }
}

double dot_product(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

void matrix_vector_product(vector<vector<double>> A, vector<double> x, vector<double>& y) {
    int n = A.size();
	#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = 0;
        for (int j = 0; j < n; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
}

int main() {
    // generate a random matrix A and vector b
    // const int n = 4096;
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<> dis(0.0, 1.0);
    // vector<vector<double>> A(n, vector<double>(n));
    // vector<double> b(n);
    // for (int i = 0; i < n; i++) {
    //     b[i] = dis(gen);
    //     for (int j = 0; j < n; j++) {
    //         A[i][j] = dis(gen);
    //     }
    // }
    const int n = 5;
    vector<double> b{1, 2, 0, 0, 5};
    vector<vector<double>> A{{1, 2, 0}, {0, 0, 4}, {4, 5, 1}, {0, 1, 0}, {0, 0, 1}};


    vector<double> x(n, 1); // initial guess for x
    vector<double> r(n); // residual
    vector<double> p(n); // search direction
    vector<double> Ap(n); // A*p

    // compute initial residual and search direction
    matrix_vector_product(A, x, Ap);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }

    int iter = 0;
    double gflops = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (iter < max_iter) {
        // compute alpha
        matrix_vector_product(A, p, Ap);
        double alpha = dot_product(r, r) / dot_product(p, Ap);

        // update x and residual
		#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // check if solution is within tolerance
        double res_norm = sqrt(dot_product(r, r));
        if (res_norm < tol) {
            break;
        }

        // compute beta and update search direction
        double beta = dot_product(r, r) / dot_product(p, p);
		#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        iter++;

        gflops += (2 * n * n + 3 * n) / 1e9;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    double gflops_per_second = gflops / elapsed_seconds;


    // print solution
    std::cout << "Solution:\n";
    for (int i = 0; i < n; i++) {
        std::cout << "x[" << i << "] = " << x[i] << "\n";
    }

    std::cout << "Gflops: " << gflops_per_second;

    return 0;
}
