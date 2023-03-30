#pragma once
#include <math.h>
#include <assert.h>
#include <stdexcept>
#include <algorithm>
#include <vector>

#include "random.hh"
#include "rational.hh"
#include "vector.hh"

namespace math
{

    template <typename VectorIn, typename VectorOut>
    void power_norm(VectorIn &input, int length, double power, VectorOut &output)
    {
        double sum = 0;
        for (int i = 0; i < length; ++i)
        {
            double x = std::pow(input[i], power);
            output[i] = x;
            sum += x;
        }
        for (int i = 0; i < length; ++i)
        {
            output[i] = output[i] / sum;
        }
    }

    template <typename Vector>
    void print(Vector &input, int length)
    {
        for (int i = 0; i < length; ++i)
        {
            std::cout << input[i] << ", ";
        }
        std::cout << std::endl;
    }
}

namespace Linear
{

    template <typename T, int size>
    class Matrix
    {
    public:
        std::array<T, size * size> data;
        int rows, cols;
        Matrix(){};
        Matrix(int rows, int cols) : rows(rows), cols(cols) {}

        T &get(int i, int j)
        {
            return data[i * cols + j];
        }

        void print()
        {
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    std::cout << get(i, j) << ", ";
                }
                std::cout << std::endl;
            }
        }

        void fill(int rows, int cols) {}

        void fill(int rows, int cols, T value)
        {
            std::fill(data.begin(), data.begin() + rows * cols, value);
        }
    };

    template <typename T>
    class MatrixVector
    {
    public:
        std::vector<T> data;
        int rows, cols;

        MatrixVector(){};
        MatrixVector(int rows, int cols) : data(std::vector<T>(rows * cols)), rows(rows), cols(cols)
        {
        }

        T &get(int i, int j)
        {
            return data[i * cols + j];
        }

        void print()
        {
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    std::cout << data[i][j] << ", ";
                }
                std::cout << std::endl;
            }
        }

        void fill(int rows, int cols) {
            data.resize(rows * cols);
        }

        void fill(int rows, int cols, T value)
        {
            const int n = rows * cols;
            data.resize(n);
            std::fill(data.begin(), data.begin() + n, value);
        }
    };

    template <class TypeList>
    typename TypeList::Real exploitability(
        typename TypeList::MatrixReal &row_payoff_matrix,
        typename TypeList::MatrixReal &col_payoff_matrix,
        typename TypeList::VectorReal &row_strategy,
        typename TypeList::VectorReal &col_strategy)
    {
        const int rows = row_payoff_matrix.rows;
        const int cols = row_payoff_matrix.cols;

        typename TypeList::Real row_payoff = 0, col_payoff = 0;
        typename TypeList::VectorReal row_response, col_response;
        row_response.fill(rows, 0);
        col_response.fill(cols, 0); // TODO maybe replace this with just a constructor
        for (int row_idx = 0; row_idx < rows; ++row_idx)
        {
            for (int col_idx = 0; col_idx < cols; ++col_idx)
            {
                const typename TypeList::Real u = row_payoff_matrix.get(row_idx, col_idx) * col_strategy[col_idx];
                const typename TypeList::Real v = col_payoff_matrix.get(row_idx, col_idx) * row_strategy[row_idx];
                row_payoff += u * row_strategy[row_idx];
                col_payoff += v * col_strategy[col_idx];
                row_response[row_idx] += u;
                col_response[col_idx] += v;
            }
        }

        typename TypeList::Real row_best_response = row_response[0], col_best_response = col_response[0];
        for (int row_idx = 1; row_idx < rows; ++row_idx)
        {
            if (row_response[row_idx] > row_best_response)
            {
                row_best_response = row_response[row_idx];
            }
        }
        for (int col_idx = 1; col_idx < cols; ++col_idx)
        {
            if (col_response[col_idx] > col_best_response)
            {
                col_best_response = col_response[col_idx];
            }
        }

        return (row_best_response - row_payoff) + (col_best_response - col_payoff);
    }
}