#pragma once

#include <types/types.h>

#include "../../extern/lrslib/include/lib.h"

namespace LRSNash {
// Solve matrix of mpq_class
template <template <typename...> typename Vector, template <typename...> typename Matrix,
          template <typename> typename Value>
Value<mpq_class> solve(const Matrix<Value<mpq_class>> &payoff_matrix, Vector<mpq_class> &row_strategy,
                       Vector<mpq_class> &col_strategy) {
    const size_t rows = payoff_matrix.rows;
    const size_t cols = payoff_matrix.cols;
    const size_t entries = rows * cols;
    std::vector<const mpq_t *> rpd{entries}, cpd{entries};

    for (size_t i = 0; i < entries; ++i) {
        rpd[i] = reinterpret_cast<mpq_t *>(&payoff_matrix[i].row_value);
        cpd[i] = reinterpret_cast<mpq_t *>(&payoff_matrix[i].col_value);
    }

    mpz_t *row_solution_data = alloc(rows + 2);
    mpz_t *col_solution_data = alloc(cols + 2);

    solve_gmp_pointer(rows, cols, rpd.data(), cpd.data(), row_solution_data, col_solution_data);

    mpz_class row_den{row_solution_data[0]}, col_den{col_solution_data[0]};
    for (int row_idx = 0; row_idx < rows; ++row_idx) {
        row_strategy[row_idx] = mpq_class{mpq_class{mpz_class{row_solution_data[row_idx + 1]}, row_den}};
    }
    for (int col_idx = 0; col_idx < cols; ++col_idx) {
        col_strategy[col_idx] = mpq_class{mpq_class{mpz_class{col_solution_data[col_idx + 1]}, col_den}};
    }

    mpq_class row_payoff{mpz_class{col_solution_data[cols + 1]}, col_den};
    mpq_class col_payoff{mpz_class{row_solution_data[rows + 1]}, row_den};

    dealloc(row_solution_data, rows + 2);
    dealloc(col_solution_data, cols + 2);

    return {mpq_class{row_payoff}, mpq_class{col_payoff}};
}

// Solve constant-sum (ConstantSum<1, 1>) matrix of mpq_class
template <template <typename...> typename Vector, template <typename...> typename Matrix,
          template <typename> typename Value>
    requires(Value<mpq_class>::IS_CONSTANT_SUM == true)
Value<mpq_class> solve(const Matrix<Value<mpq_class>> &payoff_matrix, Vector<mpq_class> &row_strategy,
                       Vector<mpq_class> &col_strategy) {
    const size_t rows = payoff_matrix.rows;
    const size_t cols = payoff_matrix.cols;
    const size_t entries = rows * cols;
    std::vector<const mpq_t *> rpd{entries};

    for (size_t i = 0; i < entries; ++i) {
        rpd[i] = reinterpret_cast<const mpq_t *>(&payoff_matrix[i].row_value);
    }

    mpz_t *row_solution_data = alloc(rows + 2);
    mpz_t *col_solution_data = alloc(cols + 2);

    solve_gmp_pointer_constant_sum(rows, cols, rpd.data(), row_solution_data, col_solution_data, 1, 1);

    mpz_class row_den{row_solution_data[0]}, col_den{col_solution_data[0]};
    row_strategy.resize(rows);
    col_strategy.resize(cols);
    for (int row_idx = 0; row_idx < rows; ++row_idx) {
        row_strategy[row_idx] = mpq_class{mpz_class{row_solution_data[row_idx + 1]}, row_den};
    }
    for (int col_idx = 0; col_idx < cols; ++col_idx) {
        col_strategy[col_idx] = mpq_class{mpz_class{col_solution_data[col_idx + 1]}, col_den};
    }

    mpq_class row_payoff{mpz_class{col_solution_data[cols + 1]}, col_den};
    mpq_class col_payoff{mpz_class{row_solution_data[rows + 1]}, row_den};

    dealloc(row_solution_data, rows + 2);
    dealloc(col_solution_data, cols + 2);

    row_payoff.canonicalize();

    return {row_payoff};
}

// AB refactor temp method
template <template <typename...> typename Vector>
mpq_class solve(const size_t rows, const size_t cols, const Vector<mpq_class> &payoff_matrix, Vector<mpq_class> &row_strategy,
                Vector<mpq_class> &col_strategy) {
    const size_t entries = rows * cols;
    std::vector<const mpq_t *> rpd{entries};

    for (size_t i = 0; i < entries; ++i) {
        rpd[i] = reinterpret_cast<const mpq_t *>(&payoff_matrix[i]);
    }

    mpz_t *row_solution_data = alloc(rows + 2);
    mpz_t *col_solution_data = alloc(cols + 2);

    solve_gmp_pointer_constant_sum(rows, cols, rpd.data(), row_solution_data, col_solution_data, 1, 1);

    mpz_class row_den{row_solution_data[0]}, col_den{col_solution_data[0]};
    row_strategy.resize(rows);
    col_strategy.resize(cols);
    for (int row_idx = 0; row_idx < rows; ++row_idx) {
        row_strategy[row_idx] = mpq_class{mpz_class{row_solution_data[row_idx + 1]}, row_den};
        row_strategy[row_idx].canonicalize();
    }
    for (int col_idx = 0; col_idx < cols; ++col_idx) {
        col_strategy[col_idx] = mpq_class{mpz_class{col_solution_data[col_idx + 1]}, col_den};
        col_strategy[col_idx].canonicalize();
    }

    mpq_class row_payoff{mpz_class{col_solution_data[cols + 1]}, col_den};
    mpq_class col_payoff{mpz_class{row_solution_data[rows + 1]}, row_den};

    dealloc(row_solution_data, rows + 2);
    dealloc(col_solution_data, cols + 2);

    row_payoff.canonicalize();

    return row_payoff;
}

// Solve for everything else, mostly for doubles
template <template <typename...> typename Vector, template <typename...> typename Matrix,
          template <typename> typename Value, typename Real>
    requires(std::is_same_v<Real, mpq_class> == false)
Value<Real> solve(const Matrix<Value<Real>> &payoff_matrix, Vector<Real> &row_strategy, Vector<Real> &col_strategy,
                  int den = 100) {
    const size_t rows = payoff_matrix.rows;
    const size_t cols = payoff_matrix.cols;
    const size_t entries = rows * cols;

    const Real min = payoff_matrix.min();
    const Real max = payoff_matrix.max();
    const Real range{max == min ? Real{1} : max - min};

    long *payoff_data = new long[2 * entries];

    for (size_t i = 0; i < entries; ++i) {
        const Value<Real> &value = payoff_matrix[i];
        double a{(value.get_row_value() - min) / range * static_cast<Real>(den)};
        double b{(value.get_col_value() - min) / range * static_cast<Real>(den)};
        payoff_data[2 * i] = ceil(a);
        payoff_data[2 * i + 1] = ceil(b);
    }

    mpz_t *row_solution_data = alloc(rows + 2);
    mpz_t *col_solution_data = alloc(cols + 2);
    const mpz_t *const_row_solution_data = row_solution_data;
    const mpz_t *const_col_solution_data = col_solution_data;

    solve_gmp_float(rows, cols, payoff_data, den, row_solution_data, col_solution_data);

    mpz_class row_den{const_row_solution_data[0]}, col_den{const_col_solution_data[0]};
    row_strategy.resize(rows);
    col_strategy.resize(cols);
    for (int row_idx = 0; row_idx < rows; ++row_idx) {
        row_strategy[row_idx] =
            Real{static_cast<Real>(mpq_class{mpz_class{const_row_solution_data[row_idx + 1]}, row_den}.get_d())};
    }
    for (int col_idx = 0; col_idx < cols; ++col_idx) {
        col_strategy[col_idx] =
            Real{static_cast<Real>(mpq_class{mpz_class{const_col_solution_data[col_idx + 1]}, col_den}.get_d())};
    }

    Real row_payoff{static_cast<Real>(mpq_class{mpz_class{const_col_solution_data[cols + 1]}, col_den}.get_d())};
    Real col_payoff{static_cast<Real>(mpq_class{mpz_class{const_row_solution_data[rows + 1]}, row_den}.get_d())};
    row_payoff = row_payoff * range + min;
    col_payoff = col_payoff * range + min;
    dealloc(row_solution_data, rows + 2);
    dealloc(col_solution_data, cols + 2);
    delete[] payoff_data;

    if constexpr (Value<Real>::IS_CONSTANT_SUM == true) {
        return {row_payoff};
    } else {
        return {row_payoff, col_payoff};
    }
}

}; // End namespace LRSNash