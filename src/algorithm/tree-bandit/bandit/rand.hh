#pragma once

#include <libsurskit/math.hh>

/*
Minimal model for benchmarking purposes (Test speed of state and tree structure)
*/

template <class Model>
class Rand : public AbstractAlgorithm<Model>
{

public:
    struct MatrixStats;
    struct ChanceStats;
    struct Types : AbstractAlgorithm<Model>::Types
    {
        using MatrixStats = Rand::MatrixStats;
        using ChanceStats = Rand::ChanceStats;
    };
    struct MatrixStats
    {
        int rows, cols;
    };

    struct ChanceStats
    {
    };
    struct Outcome
    {
    };

    Rand() {}

    friend std::ostream &operator<<(std::ostream &os, const Rand &session)
    {
        os << "Rand";
        return os;
    }

    void get_empirical_strategies(
        MatrixStats &stats,
        typename Types::VectorReal &row_strategy,
        typename Types::VectorReal &col_strategy)
    {
    }

    void get_empirical_values(
        MatrixStats &stats,
        typename Types::Real &row_value,
        typename Types::Real &col_value)
    {
    }

    void get_refined_strategies(
        MatrixStats &stats,
        typename Types::VectorReal &row_strategy,
        typename Types::VectorReal &col_strategy)
    {
    }

    void get_refined_values(
        MatrixStats &stats,
        typename Types::Real &row_value,
        typename Types::Real &col_value)
    {
    }

protected:
    void initialize_stats(
        int iterations,
        typename Types::State &state,
        typename Types::Model &model,
        MatrixStats &stats)
    {
    }

    void expand(
        MatrixStats& stats,
        typename Types::ModelOutput &inference)
    {
        stats.rows = inference.row_policy.size();
        stats.cols = inference.col_policy.size();
    }

    void select(
        typename Types::PRNG &device,
        MatrixStats &stats,
        typename Types::Outcome &outcome)
    {
        const int row_idx = device.random_int(stats.rows);
        const int col_idx = device.random_int(stats.cols);
        outcome.row_idx = row_idx;
        outcome.col_idx = col_idx;
    }

    void update_matrix_stats(
        MatrixStats &stats,
        typename Types::Outcome &outcome)
    {
    }

    void update_chance_stats(
        ChanceStats &stats,
        typename Types::Outcome &outcome)
    {
    }

    void update_matrix_stats(
        MatrixStats &stats,
        typename Types::Outcome &outcome,
        typename Types::Real learning_rate)
    {
    }

    void update_chance_stats(
        ChanceStats &stats,
        typename Types::Outcome &outcome,
        typename Types::Real learning_rate)
    {
    }

    void get_policy(
        MatrixStats &stats,
        typename Types::VectorReal &row_policy,
        typename Types::VectorReal &col_policy)
    {
        // const int rows = stats.row_gains.size();
        // const int cols = stats.col_gains.size();
        // row_policy.fill(rows);
        // col_policy.fill(cols);
        // if (rows == 1)
        // {
        //     row_policy[0] = 1;
        // }
        // else
        // {
        //     const typename Types::Real eta{gamma / static_cast<typename Types::Real>(rows)};
        //     softmax(row_policy, stats.row_gains, rows, eta);
        //     for (int row_idx = 0; row_idx < rows; ++row_idx)
        //     {
        //         row_policy[row_idx] = (1.0 - gamma) * row_policy[row_idx] + eta;
        //     }
        // }
        // if (cols == 1)
        // {
        //     col_policy[0] = 1;
        // }
        // else
        // {
        //     const typename Types::Real eta{gamma / static_cast<typename Types::Real>(cols)};
        //     softmax(col_policy, stats.col_gains, cols, eta);
        //     for (int col_idx = 0; col_idx < cols; ++col_idx)
        //     {
        //         col_policy[col_idx] = (1.0 - gamma) * col_policy[col_idx] + eta;
        //     }
        // }
    }
};