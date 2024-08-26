#pragma once

#include <libpinyon/math.h>
#include <model/model.h>

/*
Minimal model for benchmarking purposes (Test speed of state and tree structure)
*/

template <IsValueModelTypes Types>
struct Rand : Types {
    struct MatrixStats {
        int rows, cols;
    };
    struct ChanceStats {};
    struct Outcome {};

    class BanditAlgorithm {
       public:
        friend std::ostream &operator<<(std::ostream &os, const BanditAlgorithm &search) {
            os << "Rand";
            return os;
        }

        // TODO fill these out with something
        void get_empirical_strategies(const MatrixStats &stats, Types::VectorReal &row_strategy,
                                      Types::VectorReal &col_strategy) const {}

        void get_empirical_value(const MatrixStats &stats, Types::Value &value) const {}

        void get_refined_strategies(const MatrixStats &stats, Types::VectorReal &row_strategy,
                                    Types::VectorReal &col_strategy) const {}

        void get_refined_value(const MatrixStats &stats, Types::Value &value) const {}

        // protected:
        void initialize_stats(int iterations, Types::State &state, Types::Model &model, MatrixStats &stats) const {}

        void expand(MatrixStats &stats, const size_t &rows, const size_t &cols,
                    const Types::ModelOutput &output) const {
            stats.rows = rows;
            stats.cols = cols;
        }

        void select(Types::PRNG &device, const MatrixStats &stats, Outcome &outcome) const {
            outcome.row_idx = device.random_int(stats.rows);
            outcome.col_idx = device.random_int(stats.cols);
        }

        void update_matrix_stats(MatrixStats &stats, const Outcome &outcome) const {}

        void update_chance_stats(ChanceStats &stats, const Outcome &outcome) const {}

        void select(Types::PRNG &device, const MatrixStats &stats, Outcome &outcome, Types::Mutex &mutex) const {
            outcome.row_idx = device.random_int(stats.rows);
            outcome.col_idx = device.random_int(stats.cols);
        }

        void update_matrix_stats(MatrixStats &stats, const Outcome &outcome, Types::Mutex &mutex) const {}

        void update_chance_stats(ChanceStats &stats, const Outcome &outcome, Types::Mutex &mutex) const {}
    };
};
