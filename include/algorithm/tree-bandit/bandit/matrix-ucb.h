#pragma once

#include <assert.h>

#include <algorithm/algorithm.h>
#include <concepts>
#include <libpinyon/lrslib.h>
#include <libpinyon/math.h>
#include <tree/tree.h>
#include <types/matrix.h>

/*
MatrixUCB (Uses usual UCB formula, no time estimates)
*/

template <IsValueModelTypes Types>
struct MatrixUCB : Types {
    using Real = typename Types::Real;
    using MatrixPairReal = typename Types::template Matrix<PairReal<Real>>;

   public:
    struct MatrixStats {
        struct Data {
            Types::Value value;
            int visits;
        };
        int total_visits{};
        DataMatrix<Data> data_matrix;
        Types::VectorReal row_strategy;
        Types::VectorReal col_strategy;
    };

    struct ChanceStats {};

    struct Outcome {
        int row_idx, col_idx;
        Types::Value value;
        Types::Real row_mu, col_mu;
    };

    class BanditAlgorithm {
       public:
        BanditAlgorithm() {}

        BanditAlgorithm(Types::Real) {}

        BanditAlgorithm(Types::Real c_uct, Types::Real expl_threshold) : c_uct(c_uct), expl_threshold(expl_threshold) {}

        friend std::ostream &operator<<(std::ostream &os, const BanditAlgorithm &session) {
            os << "MatrixUCB; c_uct: " << session.c_uct << ", expl_threshold: " << session.expl_threshold;
            return os;
        }

        const Types::Real c_uct{2};
        const Types::Real expl_threshold{.005};
        // bool require_interior = false;

        void initialize_stats(int playouts, Types::State &state, Types::Model &model, MatrixStats &stats) {}

        void get_empirical_strategies(const MatrixStats &stats, Types::VectorReal &row_strategy,
                                      Types::VectorReal &col_strategy) const {
            const int rows = stats.data_matrix.rows;
            const int cols = stats.data_matrix.cols;
            row_strategy.resize(rows);
            col_strategy.resize(cols);
            for (int row_idx = 0; row_idx < rows; ++row_idx) {
                for (int col_idx = 0; col_idx < cols; ++col_idx) {
                    const int n = stats.data_matrix.get(row_idx, col_idx).visits;
                    row_strategy[row_idx] += n;
                    col_strategy[col_idx] += n;
                }
            }
            math::power_norm(row_strategy, rows, 1, row_strategy);
            math::power_norm(col_strategy, cols, 1, col_strategy);
        }

        void get_refined_strategies(const MatrixStats &stats, Types::VectorReal &row_strategy,
                                    Types::VectorReal &col_strategy) {
            const auto value_matrix = get_ev_matrix(stats);
            LRSNash::solve(value_matrix, row_strategy, col_strategy);
        }

        void get_empirical_value(const MatrixStats &stats, Types::Value value) const
        {
            // TODO - added temp flag to search model to ignore this
        };

        void expand(MatrixStats &stats, const size_t &rows, const size_t &cols,
                    const Types::ModelOutput &output) const {  // matrix_node->is_expanded = true;
            stats.data_matrix.fill(rows, cols, {});
            // uniform initialization of stats.strategies
            stats.row_strategy.resize(rows, 1 / static_cast<Real>(rows));
            stats.col_strategy.resize(cols, 1 / static_cast<Real>(cols));
        }

        void select(Types::PRNG &device, MatrixStats &stats, Outcome &outcome) const {
            const size_t rows = stats.data_matrix.rows;
            const size_t cols = stats.data_matrix.cols;
            MatrixPairReal ucb_matrix{rows, cols};
            get_ucb_matrix(stats, ucb_matrix);
            typename Types::VectorReal &row_strategy = stats.row_strategy;
            typename Types::VectorReal &col_strategy = stats.col_strategy;
            Real expl = math::exploitability(ucb_matrix, row_strategy, col_strategy);
            if (expl > expl_threshold) {
                LRSNash::solve(ucb_matrix, row_strategy, col_strategy);
            }
            outcome.row_idx = device.sample_pdf(row_strategy);
            outcome.col_idx = device.sample_pdf(col_strategy);
        }

        void update_matrix_stats(MatrixStats &stats, const Outcome &outcome) const {
            ++stats.total_visits;
            auto &data = stats.data_matrix.get(outcome.row_idx, outcome.col_idx);
            data.value += outcome.value;
            ++data.visits;
        }

        void update_chance_stats(const ChanceStats &stats, const Outcome &outcome) const {}

        // private:
        void get_ucb_matrix(const MatrixStats &stats, MatrixPairReal &ucb_matrix) const {
            auto &data_matrix = stats.data_matrix;
            const int rows = data_matrix.rows;
            const int cols = data_matrix.cols;
            const Real log_total = std::log(stats.total_visits);
            for (int row_idx = 0; row_idx < rows; ++row_idx) {
                for (int col_idx = 0; col_idx < cols; ++col_idx) {
                    const int entry_idx = row_idx * cols + col_idx;
                    const auto &data = data_matrix[entry_idx];
                    auto &ucb_pair = ucb_matrix[entry_idx];
                    int n = data.visits;
                    n += (n == 0);
                    const Real eta = this->c_uct * std::sqrt(log_total / n);
                    ucb_pair.row_value = data.value.get_row_value() / n + eta;
                    ucb_pair.col_value = data.value.get_col_value() / n + eta;
                }
            }
        }

        auto get_ev_matrix(const MatrixStats &stats) const {
            auto &data_matrix = stats.data_matrix;
            const int rows = data_matrix.rows;
            const int cols = data_matrix.cols;
            typename Types::MatrixValue matrix{static_cast<size_t>(rows), static_cast<size_t>(cols)};
            const Real log_total = std::log(stats.total_visits);
            for (int row_idx = 0; row_idx < rows; ++row_idx) {
                for (int col_idx = 0; col_idx < cols; ++col_idx) {
                    const auto &data = data_matrix.get(row_idx, col_idx);
                    int n = data.visits;
                    n += (n == 0);
                    const Real eta = this->c_uct * std::sqrt(log_total / n);
                    matrix.get(row_idx, col_idx) = data.value.get_row_value() / n + eta;
                }
            }
            return matrix;
        }
    };
};