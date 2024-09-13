#include <concepts>
#include <vector>
#include <chrono>

#include "./types/random.h"

// template <typename T>
// concept State = requires () {
//     true;
// };

struct MoldState
{
    unsigned int depth{};
    // std::vector<int> row_actions{};
    // std::vector<int> col_actions{};
    const unsigned int row_actions;
    const unsigned int col_actions;

    MoldState(unsigned int depth = 1, unsigned int actions = 2) : depth{depth}, row_actions{actions}, col_actions{actions}
    {
    }

    bool is_terminal() const
    {
        return depth == 0;
    }

    void get_actions() const
    {
    }

    float get_payoff() const
    {
        return .5;
    }

    void apply_actions(int, int)
    {
        --depth;
    }
};

// template <typename

template <typename State>
void apply_uniform_actions(State &&);

template <>
void apply_uniform_actions<int>(int &&) {}

template <typename PRNG>
struct MonteCarloModel
{

    PRNG device;

    // MonteCarloModel (const PRNG &device) : device{device} {}
    template <typename State>
        requires(std::is_integral_v<typename State::row_actions>)
    auto inference(State &&state)
    {
        while (!state.is_terminal())
        {
            state.apply_actions(
                device.uniform_int(state.row_actions),
                device.uniform_int(state.col_actions));
            state.get_actions();
        }

        return state.get_payoff();
    }

    template <typename State>
    auto inference(State &&state)
    {
        while (!state.is_terminal())
        {
            state.apply_actions(
                state.row_actions[device.uniform_int(state.row_actions.size())],
                state.col_actions[device.uniform_int(state.col_actions.size())]);
            state.get_actions();
        }

        return state.get_payoff();
    }
};

template <typename Gamma = float>
struct Exp3
{
    Gamma gamma;
    Gamma one_minus_gamma;

    Exp3() : gamma{.1}, one_minus_gamma{.9} {}

    template <typename Options, typename Stats, typename PRNG, typename Outcome>
    auto select(PRNG &device, Stats &stats) const
    {
        const size_t rows = stats.row_gains.size();
        const size_t cols = stats.col_gains.size();

        decltype(stats.row_gains) row_forecast{rows};
        decltype(stats.col_gains) col_forecast{cols};

        if (rows == 1)
        {
            row_forecast[0] = 1;
        }
        else
        {
            const Gamma eta{gamma / static_cast<Gamma>(rows)};
            softmax(row_forecast, stats.row_gains, rows, eta);
            std::transform(row_forecast.begin(), row_forecast.end(), row_forecast.begin(),
                           [eta, one_minus_gamma](Real value)
                           { return one_minus_gamma * value + eta; });
        }
        const int row_idx = device.sample_pdf(row_forecast);
        const int col_idx = device.sample_pdf(col_forecast);
        outcome.row_idx = row_idx;
        outcome.col_idx = col_idx;
        outcome.row_mu = row_forecast[row_idx];
        outcome.col_mu = col_forecast[col_idx];
    }

    template <typename Stats, typename Outcome, typename Real>
    void update_matrix_stats(Stats &stats, const Outcome &outcome) const
    {
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
    }

    template <typename Stats, typename ModelOutput,
              void expand(Stats &stats, const int rows, const int cols, const ModelOutput &output) const
    {
        stats.row_visits.resize(rows, 0);
        stats.col_visits.resize(cols, 0);
        stats.row_gains.resize(rows, 0);
        stats.col_gains.resize(cols, 0);
    }

private:
    template <template <typename...> Vector, typename Real>
    inline void softmax(Vector<Real> &forecast, const Vector<Real> &gains, const size_t k, Real eta) const
    {
        Real sum = 0;
        for (size_t i = 0; i < k; ++i)
        {
            const Real y{std::exp(gains[i] * eta)};
            forecast[i] = y;
            sum += y;
        }
        for (size_t i = 0; i < k; ++i)
        {
            forecast[i] /= sum;
        }
    };

    // template <typename,
};

struct TreeBandit
{

    template <typename Bandit, typename PRNG, typename State, typename Model, typename MatrixNode>
    size_t run(size_t ms, PRNG &device, const State &state, Model &model, MatrixNode &node)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        do
        {
            this->run_iteration<Bandit>(device, state, model, &node);
        } while (
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start)
                .count() < ms);
        return 0;
    }

    template <typename Bandit, typename PRNG, typename State, typename Model, typename MatrixNode, typename ModelOutput, typename Outcome>
    MatrixNode *run_iteration(
        PRNG &device,
        State &state,
        Model &model,
        MatrixNode *matrix_node,
        ModelOutput &model_output) const
    {
        if (state.is_terminal())
        {
            matrix_node->set_terminal();
            model_output.value = state.get_payoff();
            return matrix_node;
        }
        else
        {
            if (!matrix_node->is_expanded())
            {
                const size_t rows = state.row_actions.size();
                const size_t cols = state.col_actions.size();
                model.inference(std::move(state), model_output);
                matrix_node->expand(rows, cols);
                Bandit::expand(matrix_node->stats, rows, cols, model_output);
                return matrix_node;
            }
            else
            {
                Outcome outcome;
                Bandit::select(device, matrix_node->stats, outcome);

                ChanceNode *chance_node = matrix_node->access(outcome.row_idx, outcome.col_idx);

                state.apply_actions(
                    state.row_actions[outcome.row_idx],
                    state.col_actions[outcome.col_idx]);
                state.get_actions();

                MatrixNode *matrix_node_next = chance_node->access(state.get_obs());
                MatrixNode *matrix_node_leaf = run_iteration(device, state, model, matrix_node_next, model_output);
                outcome.value = model_output.value;

                this->update_matrix_stats(matrix_node->stats, outcome);
                this->update_chance_stats(chance_node->stats, outcome);
                return matrix_node_leaf;
            }
        }
    }
};

int main()
{
    MoldState state{10, 2};
    

    return 0;
}