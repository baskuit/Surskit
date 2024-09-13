#pragma once

#include <libpinyon/math.h>
#include <types/matrix.h>
#include <algorithm/algorithm.h>
#include <tree/tree.h>

/*

Exp3 but with additional statistics. Behaves the same.

*/

template <IsValueModelTypes Types>
struct Exp3Fat : Types
{

    using Real = typename Types::Real;

    struct Data
    {
        Real cum_row_value;
        double count;
        friend std::ostream &operator<<(std::ostream &os, const Data &data)
        {
            os << '(' << (data.cum_row_value / typename Types::Real{data.count}) << ", " << data.count << ')';
            return os;
        }
    };

    struct MatrixStats
    {
        Types::VectorReal row_gains;
        Types::VectorReal col_gains;
        Types::VectorInt row_visits;
        Types::VectorInt col_visits;

        int visits = 0;
        PairReal<Real> value_total{0, 0};

        DataMatrix<Data> matrix;
    };
    struct ChanceStats
    {
    };
    struct Outcome
    {
        int row_idx, col_idx;
        typename Types::Value value;
        Real row_mu, col_mu;
    };

    class BanditAlgorithm
    {
    public:
        const Real gamma{.01};
        const Real one_minus_gamma{gamma * -1 + 1};

        BanditAlgorithm() {}

        constexpr BanditAlgorithm(Real gamma) : gamma(gamma), one_minus_gamma{gamma * -1 + 1} {}

        friend std::ostream &operator<<(std::ostream &os, const BanditAlgorithm &search)
        {
            os << "Exp3Fat; gamma: " << search.gamma;
            return os;
        }

        void get_empirical_strategies(
            MatrixStats &stats,
            Types::VectorReal &row_strategy,
            Types::VectorReal &col_strategy) const
        {
            row_strategy.resize(stats.row_visits.size());
            col_strategy.resize(stats.col_visits.size());
            math::power_norm(stats.row_visits, row_strategy.size(), 1, row_strategy);
            math::power_norm(stats.col_visits, col_strategy.size(), 1, col_strategy);
        }

        void get_empirical_value(
            MatrixStats &stats,
            Types::Value &value) const
        {
            const Real den = 1 / (stats.visits + (stats.visits == 0));
            if constexpr (Types::Value::IS_CONSTANT_SUM)
            {
                value = typename Types::Value{typename Types::Real{stats.value_total.get_row_value() * den}};
            }
            else
            {
                value = typename Types::Value{stats.value_total * den};
            }
        }

        void get_refined_strategies(
            MatrixStats &stats,
            Types::VectorReal &row_strategy,
            Types::VectorReal &col_strategy) const
        {
            row_strategy.resize(stats.row_visits.size());
            col_strategy.resize(stats.col_visits.size());
            denoise(row_strategy, col_strategy);
            math::power_norm(stats.row_visits, row_strategy.size(), 1, row_strategy);
            math::power_norm(stats.col_visits, col_strategy.size(), 1, col_strategy);
        }

        void get_refined_value(
            MatrixStats &stats,
            Types::Value &value) const
        {
            get_empirical_value(stats, value);
        }

        // protected:
        void initialize_stats(
            int iterations,
            const Types::State &state,
            Types::Model &model,
            MatrixStats &stats) const
        {
        }

        void expand(
            MatrixStats &stats,
            const size_t &rows,
            const size_t &cols,
            const Types::ModelOutput &output) const
        {
            stats.row_visits.resize(rows, 0);
            stats.col_visits.resize(cols, 0);
            stats.row_gains.resize(rows, 0);
            stats.col_gains.resize(cols, 0);
            stats.matrix.fill(rows, cols);
        }

        void select(
            Types::PRNG &device,
            MatrixStats &stats,
            Outcome &outcome) const
        {
            const int rows = stats.row_gains.size();
            const int cols = stats.col_gains.size();
            const auto &one_minus_gamma = this->one_minus_gamma;
            typename Types::VectorReal row_forecast(rows);
            typename Types::VectorReal col_forecast(cols);
            if (rows == 1)
            {
                row_forecast[0] = Rational<>{1};
            }
            else
            {
                const Real eta{gamma / static_cast<Real>(rows)};
                softmax(row_forecast, stats.row_gains, rows, eta);
                std::transform(
                    row_forecast.begin(), row_forecast.begin() + rows, row_forecast.begin(),
                    [eta, one_minus_gamma](Real value)
                    { return one_minus_gamma * value + eta; });
            }
            if (cols == 1)
            {
                col_forecast[0] = Rational<>{1};
            }
            else
            {
                const Real eta{gamma / static_cast<Real>(cols)};
                softmax(col_forecast, stats.col_gains, cols, eta);
                std::transform(
                    col_forecast.begin(), col_forecast.begin() + cols, col_forecast.begin(),
                    [eta, one_minus_gamma](Real value)
                    { return one_minus_gamma * value + eta; });
            }
            const int row_idx = device.sample_pdf(row_forecast);
            const int col_idx = device.sample_pdf(col_forecast);
            outcome.row_idx = row_idx;
            outcome.col_idx = col_idx;
            outcome.row_mu = static_cast<Real>(row_forecast[row_idx]);
            outcome.col_mu = static_cast<Real>(col_forecast[col_idx]);
        }

        void update_matrix_stats(
            MatrixStats &stats,
            Outcome &outcome) const
        {
            stats.value_total += PairReal<Real>{outcome.value.get_row_value(), outcome.value.get_col_value()};
            stats.visits += 1;
            stats.row_visits[outcome.row_idx] += 1;
            stats.col_visits[outcome.col_idx] += 1;
            if ((stats.row_gains[outcome.row_idx] += outcome.value.get_row_value() / outcome.row_mu) >= 0)
            {
                const auto max = stats.row_gains[outcome.row_idx];
                for (auto &v : stats.row_gains)
                {
                    v -= max;
                }
            }
            if ((stats.col_gains[outcome.col_idx] += outcome.value.get_col_value() / outcome.col_mu) >= 0)
            {
                const auto max = stats.col_gains[outcome.col_idx];
                for (auto &v : stats.col_gains)
                {
                    v -= max;
                }
            }
            Data *data_ptr;
            Real value;
            if (outcome.row_idx > outcome.col_idx)
            {
                data_ptr = &stats.matrix.get(outcome.col_idx, outcome.row_idx);
                value = outcome.value.get_col_value();
            }
            else
            {
                data_ptr = &stats.matrix.get(outcome.row_idx, outcome.col_idx);
                value = outcome.value.get_row_value();
            }

            Data &data = *data_ptr;
            data.cum_row_value += value;
            data.count++;
        }

        void update_chance_stats(
            ChanceStats &stats,
            Outcome &outcome) const
        {
        }

        // multithreaded

        void select(
            Types::PRNG &device,
            MatrixStats &stats,
            Outcome &outcome,
            Types::Mutex &mutex) const
        {
            mutex.lock();
            typename Types::VectorReal row_forecast{stats.row_gains};
            typename Types::VectorReal col_forecast{stats.col_gains};
            mutex.unlock();
            const int rows = row_forecast.size();
            const int cols = col_forecast.size();
            const auto &one_minus_gamma = this->one_minus_gamma;

            if (rows == 1)
            {
                row_forecast[0] = Rational<>{1};
            }
            else
            {
                const Real eta{gamma / static_cast<Real>(rows)};
                softmax(row_forecast, row_forecast, rows, eta);
                std::transform(
                    row_forecast.begin(), row_forecast.begin() + rows, row_forecast.begin(),
                    [eta, one_minus_gamma](Real value)
                    { return one_minus_gamma * value + eta; });
            }
            if (cols == 1)
            {
                col_forecast[0] = Rational<>{1};
            }
            else
            {
                const Real eta{gamma / static_cast<Real>(cols)};
                softmax(col_forecast, col_forecast, cols, eta);
                std::transform(
                    col_forecast.begin(), col_forecast.begin() + cols, col_forecast.begin(),
                    [eta, one_minus_gamma](Real value)
                    { return one_minus_gamma * value + eta; });
            }
            const int row_idx = device.sample_pdf(row_forecast, rows);
            const int col_idx = device.sample_pdf(col_forecast, cols);
            outcome.row_idx = row_idx;
            outcome.col_idx = col_idx;
            outcome.row_mu = static_cast<Real>(row_forecast[row_idx]);
            outcome.col_mu = static_cast<Real>(col_forecast[col_idx]);
        }

        void update_matrix_stats(
            MatrixStats &stats,
            Outcome &outcome,
            Types::Mutex &mutex) const
        {
            mutex.lock();
            stats.value_total += outcome.value;
            stats.visits += 1;
            stats.row_visits[outcome.row_idx] += 1;
            stats.col_visits[outcome.col_idx] += 1;
            if ((stats.row_gains[outcome.row_idx] += outcome.value.get_row_value() / outcome.row_mu) >= 0)
            {
                const auto max = stats.row_gains[outcome.row_idx];
                for (auto &v : stats.row_gains)
                {
                    v -= max;
                }
            }
            if ((stats.col_gains[outcome.col_idx] += outcome.value.get_col_value() / outcome.col_mu) >= 0)
            {
                const auto max = stats.col_gains[outcome.col_idx];
                for (auto &v : stats.col_gains)
                {
                    v -= max;
                }
            }
            Data *data_ptr;
            Real value;
            if (outcome.row_idx > outcome.col_idx)
            {
                data_ptr = &stats.matrix.get(outcome.col_idx, outcome.row_idx);
                value = outcome.value.get_col_value();
            }
            else
            {
                data_ptr = &stats.matrix.get(outcome.row_idx, outcome.col_idx);
                value = outcome.value.get_row_value();
            }

            Data &data = *data_ptr;
            data.cum_row_value += value;
            data.count++;
            mutex.unlock();
        }

        void update_chance_stats(
            ChanceStats &stats,
            Outcome &outcome,
            Types::Mutex &mutex) const
        {
        }

        // off-policy

        void update_matrix_stats(
            MatrixStats &stats,
            Outcome &outcome,
            Real learning_rate) const
        {
            stats.value_total += outcome.value * learning_rate;
            stats.visits += 1;
            stats.row_visits[outcome.row_idx] += 1;
            stats.col_visits[outcome.col_idx] += 1;
            stats.row_gains[outcome.row_idx] += outcome.value.get_row_value() / outcome.row_mu * learning_rate;
            stats.col_gains[outcome.col_idx] += outcome.value.get_col_value() / outcome.col_mu * learning_rate;
        }

        void update_chance_stats(
            ChanceStats &stats,
            Outcome &outcome,
            Real learning_rate) const
        {
        }

        void get_policy(
            MatrixStats &stats,
            Types::VectorReal &row_policy,
            Types::VectorReal &col_policy) const
        {
            const int rows = stats.row_gains.size();
            const int cols = stats.col_gains.size();
            const auto &one_minus_gamma = this->one_minus_gamma;
            if (rows == 1)
            {
                row_policy[0] = 1;
            }
            else
            {
                const Real eta{gamma / static_cast<Real>(rows)};
                softmax(row_policy, stats.row_gains, rows, eta);
                std::transform(
                    row_policy.begin(), row_policy.begin() + rows, row_policy.begin(),
                    [eta, one_minus_gamma](Real value)
                    { return one_minus_gamma * value + eta; });
            }
            if (cols == 1)
            {
                col_policy[0] = 1;
            }
            else
            {
                const Real eta{gamma / static_cast<Real>(cols)};
                softmax(col_policy, stats.col_gains, cols, eta);
                std::transform(
                    col_policy.begin(), col_policy.begin() + cols, col_policy.begin(),
                    [eta, one_minus_gamma](Real value)
                    { return one_minus_gamma * value + eta; });
            }
        }

    private:
        inline void softmax(
            Types::VectorReal &forecast,
            Types::VectorReal &gains,
            const size_t k,
            Real eta) const
        {
            Real sum = 0;
            for (size_t i = 0; i < k; ++i)
            {
                const Real y{std::exp(static_cast<double>(gains[i] * eta))};
                forecast[i] = y;
                sum += y;
            }
            for (size_t i = 0; i < k; ++i)
            {
                forecast[i] /= sum;
            }
        };

        inline void denoise(
            Types::VectorReal &row_strategy,
            Types::VectorReal &col_strategy) const
        {
            const int rows = row_strategy.size();
            const int cols = col_strategy.size();
            const auto &one_minus_gamma = this->one_minus_gamma;
            if (rows > 1)
            {
                const Real eta{gamma / static_cast<Real>(rows)};
                std::transform(row_strategy.begin(), row_strategy.begin() + rows, row_strategy.begin(),
                               [eta, one_minus_gamma](Real value)
                               { return (value - eta) / one_minus_gamma; });
            }
            if (cols > 1)
            {
                const Real eta{gamma / static_cast<Real>(cols)};
                std::transform(col_strategy.begin(), col_strategy.begin() + cols, col_strategy.begin(),
                               [eta, one_minus_gamma](Real value)
                               { return (value - eta) / one_minus_gamma; });
            }
        } // TODO can produce negative values but this shouldnt cause problems.
    };
};