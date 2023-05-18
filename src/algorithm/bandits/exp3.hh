#pragma once

#include "bandit.hh"

#include "../../libsurskit/math.hh"

/*
Exp3
*/

template <class Model, template <class _Model, class _BanditAlgorithm, class Outcome> class _TreeBandit>
class Exp3 : public _TreeBandit<Model, Exp3<Model, _TreeBandit>, ChoicesOutcome<Model>>
{
public:
    struct MatrixStats;
    struct ChanceStats;
    struct Types : _TreeBandit<Model, Exp3<Model, _TreeBandit>,  ChoicesOutcome<Model>>::Types
    {
        using MatrixStats = Exp3::MatrixStats;
        using ChanceStats = Exp3::ChanceStats;
    };
    struct MatrixStats : _TreeBandit<Model, Exp3<Model, _TreeBandit>, ChoicesOutcome<Model>>::MatrixStats
    {
        typename Types::VectorReal row_gains;
        typename Types::VectorReal col_gains;
        typename Types::VectorInt row_visits;
        typename Types::VectorInt col_visits;

        int visits = 0;
        typename Types::Real row_value_total = 0;
        typename Types::Real col_value_total = 0;
    };

    struct ChanceStats : _TreeBandit<Model, Exp3<Model, _TreeBandit>, ChoicesOutcome<Model>>::ChanceStats
    {
        int visits = 0;
        typename Types::Real row_value_total = 0;
        typename Types::Real col_value_total = 0;
    };

    typename Types::Real gamma = .01;

    Exp3() {}

    Exp3(typename Types::Real gamma) : gamma(gamma) {}

    friend std::ostream &operator<<(std::ostream &os, const Exp3 &session)
    {
        os << "Exp3p; gamma: " << session.gamma;
        return os;
    }

    void get_empirical_strategies(
        MatrixNode<Exp3> *matrix_node,
        typename Types::VectorReal &row_strategy,
        typename Types::VectorReal &col_strategy)
    {
        row_strategy.fill(matrix_node->stats.row_visits.size());
        col_strategy.fill(matrix_node->stats.col_visits.size());
        // denoise?
        math::power_norm(matrix_node->stats.row_visits, row_strategy.size(), 1, row_strategy);
        math::power_norm(matrix_node->stats.col_visits, col_strategy.size(), 1, col_strategy);
    }

    void get_empirical_values(
        MatrixNode<Exp3> *matrix_node,
        typename Types::Real &row_value,
        typename Types::Real &col_value)
    {
        auto stats = matrix_node->stats;
        const typename Types::Real den = 1 / (stats.total_visits + (stats.total_visits == 0));
        row_value = stats.row_value_total * den;
        col_value = stats.col_value_total * den;
    }

    void initialize_stats(
        int iterations,
        typename Types::State &state,
        typename Types::Model &model,
        MatrixNode<Exp3> *matrix_node)
    {
        // i.e. time, so not used
    }

    void expand(
        typename Types::State &state,
        typename Types::Model &model,
        MatrixNode<Exp3> *matrix_node)
    {
        matrix_node->stats.row_visits.fill(state.row_actions.size(), 0);
        matrix_node->stats.col_visits.fill(state.col_actions.size(), 0);
        matrix_node->stats.row_gains.fill(state.row_actions.size(), 0);
        matrix_node->stats.col_gains.fill(state.col_actions.size(), 0);
    }

    void select(
        typename Types::PRNG &device,
        MatrixNode<Exp3> *matrix_node,
        typename Types::Outcome &outcome)
    {
        /*
        Softmaxing of the gains to produce forecasts/strategies for the row and col players.
        The constants eta, gamma, beta are from (arXiv:1204.5721), Theorem 3.3.
        */
        const int rows = matrix_node->row_actions.size();
        const int cols = matrix_node->col_actions.size();
        typename Types::VectorReal row_forecast(rows);
        typename Types::VectorReal col_forecast(cols);
        if (rows == 1)
        {
            row_forecast[0] = 1;
        }
        else
        {
            const typename Types::Real eta = gamma / rows;
            softmax(row_forecast, matrix_node->stats.row_gains, rows, eta);
            for (int row_idx = 0; row_idx < rows; ++row_idx)
            {
                row_forecast[row_idx] = (1 - gamma) * row_forecast[row_idx] + eta;
            }
        }
        if (cols == 1)
        {
            col_forecast[0] = 1;
        }
        else
        {
            const typename Types::Real eta = gamma / cols;
            softmax(col_forecast, matrix_node->stats.col_gains, cols, eta);
            for (int col_idx = 0; col_idx < cols; ++col_idx)
            {
                col_forecast[col_idx] = (1 - gamma) * col_forecast[col_idx] + eta;
            }
        }
        const int row_idx = device.sample_pdf(row_forecast, rows);
        const int col_idx = device.sample_pdf(col_forecast, cols);
        outcome.row_idx = row_idx;
        outcome.col_idx = col_idx;
        outcome.row_mu = row_forecast[row_idx];
        outcome.col_mu = col_forecast[col_idx];
    }

    void update_matrix_node(
        MatrixNode<Exp3> *matrix_node,
        typename Types::Outcome &outcome)
    {
        matrix_node->stats.row_value_total += outcome.row_value;
        matrix_node->stats.col_value_total += outcome.col_value;
        matrix_node->stats.visits += 1;
        matrix_node->stats.row_visits[outcome.row_idx] += 1;
        matrix_node->stats.col_visits[outcome.col_idx] += 1;
        matrix_node->stats.row_gains[outcome.row_idx] += outcome.row_value / outcome.row_mu;
        matrix_node->stats.col_gains[outcome.col_idx] += outcome.col_value / outcome.col_mu;
    }

    void update_chance_node(
        ChanceNode<Exp3> *chance_node,
        typename Types::Outcome &outcome)
    {
        chance_node->stats.row_value_total += outcome.row_value;
        chance_node->stats.col_value_total += outcome.col_value;
        chance_node->stats.visits += 1;
    }

private:
    inline void softmax(
        typename Types::VectorReal &forecast,
        typename Types::VectorReal &gains,
        int k,
        typename Types::Real eta)
    {
        /*
        Softmax helper function with logit scaling.
        */
        typename Types::Real max = 0;
        for (int i = 0; i < k; ++i)
        {
            typename Types::Real x = gains[i];
            if (x > max)
            {
                max = x;
            }
        }
        typename Types::Real sum = 0;
        for (int i = 0; i < k; ++i)
        {
            gains[i] -= max;
            typename Types::Real x = gains[i];
            typename Types::Real y = std::exp(x * eta);
            forecast[i] = y;
            sum += y;
        }
        for (int i = 0; i < k; ++i)
        {
            forecast[i] /= sum;
        }
    };
};