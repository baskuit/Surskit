#pragma once

#include <math.h>

#include <cstdint>
#include <types/random.h>
#include <types/rational.h>

namespace math {

template <typename T>
void canonicalize(T &) {}

void canonicalize(mpq_class &t) { t.canonicalize(); }

template <typename T>
void canonicalize(Rational<T> &t) {
    t.canonicalize();
}

float to_float(mpq_class x) { return static_cast<float>(x.get_d()); }

template <typename Real>
float to_float(Real x) {
    return static_cast<float>(x);
}

double to_double(mpq_class x) { return x.get_d(); }

template <typename Real>
float to_double(Real x) {
    return static_cast<double>(x);
}

template <template <typename...> typename VectorIn, typename InT, template <typename...> typename VectorOut,
          typename OutT>
void power_norm(const VectorIn<InT> &input, int length, float power, VectorOut<OutT> &output) {
    float sum = 0;
    for (int i = 0; i < length; ++i) {
        float x = std::pow(input[i], power);
        output[i] = OutT{x};
        sum += x;
    }
    for (int i = 0; i < length; ++i) {
        output[i] = output[i] / static_cast<OutT>(sum);
    }
}

template <typename VectorIn>
void power_norm(VectorIn &input, float power = 1.0) {
    float sum = 0;
    const size_t length = input.size();
    for (int i = 0; i < length; ++i) {
        float x = std::pow(input[i], power);
        input[i] = x;
        sum += x;
    }
    for (int i = 0; i < length; ++i) {
        input[i] = input[i] / sum;
    }
}

template <template <typename...> typename Vector, typename T>
void print(const Vector<T> &input) {
    for (int i = 0; i < input.size(); ++i) {
        std::cout << input[i] << ", ";
    }
    std::cout << std::endl;
}

template <typename Real, typename Value, template <typename...> typename Vector, template <typename> typename Matrix>
Real exploitability(const Matrix<Value> &value_matrix, const Vector<Real> &row_strategy,
                    const Vector<Real> &col_strategy) {
    const size_t rows = value_matrix.rows;
    const size_t cols = value_matrix.cols;

    Real row_payoff{Rational<>{0}}, col_payoff{Rational<>{0}};
    Vector<Real> row_response, col_response;
    row_response.resize(rows);  // TODO use pinyon interface
    col_response.resize(cols);

    size_t data_idx = 0;
    for (int row_idx = 0; row_idx < rows; ++row_idx) {
        for (int col_idx = 0; col_idx < cols; ++col_idx) {
            const Value &value = value_matrix[data_idx];
            const Real u{col_strategy[col_idx] * value.get_row_value()};
            const Real v{row_strategy[row_idx] * value.get_col_value()};
            row_payoff += u * row_strategy[row_idx];
            col_payoff += v * col_strategy[col_idx];
            row_response[row_idx] += u;
            col_response[col_idx] += v;
            ++data_idx;
        }
    }

    Real row_best_response{*std::max_element(row_response.begin(), row_response.end())};
    Real col_best_response{*std::max_element(col_response.begin(), col_response.end())};

    return Real{row_best_response - row_payoff + col_best_response - col_payoff};
}

}  // namespace math
