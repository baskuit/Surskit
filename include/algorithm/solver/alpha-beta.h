#pragma once

#include <libpinyon/math.h>
#include <libpinyon/lrslib.h>
#include <types/matrix.h>
#include <algorithm/algorithm.h>
#include <tree/tree.h>

template <IsSingleModelTypes Types, template <typename...> typename NodePair = DefaultNodes>
struct AlphaBeta : Types
{
    using Real = Types::Real;

    struct Data
    {
        Types::Prob unexplored{1};
        Real alpha_explored{0}, beta_explored{0};
        int next_chance_idx = 0;
        std::vector<typename Types::Obs> chance_actions{};

        friend std::ostream &operator<<(std::ostream &os, const Data &data)
        {
            if constexpr (std::is_same_v<typename Types::Real, mpq_class>)
            {
                os << '(' << data.alpha_explored.get_d() << " " << data.beta_explored.get_d() << " " << data.unexplored.get_d() << ")";
            }
            else
            {
                os << '(' << data.alpha_explored << " " << data.beta_explored << " " << data.unexplored << ")";
            }
            return os;
        }
    };
    struct MatrixStats
    {
        DataMatrix<Data> chance_data_matrix{};
        Types::VectorReal row_solution{}, col_solution{};
        unsigned int depth = 0;

        int row_pricipal_idx = 0, col_pricipal_idx = 0;
        std::vector<int> I{}, J{};

        Types::Value solved_value;
    };
    struct ChanceStats
    {
    };
    using MatrixNode = NodePair<Types, MatrixStats, ChanceStats>::MatrixNode;
    using ChanceNode = NodePair<Types, MatrixStats, ChanceStats>::ChanceNode;

    class Search
    {
    public:
        NodePair<
            Types,
            typename FullTraversal<Types>::MatrixStats,
            typename FullTraversal<Types>::ChanceStats>::MatrixNode *teacher;

        const Real min_val{0}; // don't need to use the Game values if you happen to know that State's
        const Real max_val{1};

        Search() {}

        Search(Real min_val, Real max_val) : min_val{min_val}, max_val{max_val} {}

        auto run(
            const size_t max_depth,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode &root) const
        {
            auto state_copy = state;
            return double_oracle(max_depth, device, state_copy, model, &root, min_val, max_val);
        }

        std::pair<Real, Real>
        double_oracle(
            const size_t max_depth,
            Types::PRNG &device,
            Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            Real alpha,
            Real beta) const
        {
            if (state.is_terminal())
            {
                matrix_node->set_terminal();
                const typename Types::Value payoff = state.get_payoff();
                matrix_node->stats.solved_value = payoff;
                return {payoff.get_row_value(), payoff.get_row_value()};
            }

            state.get_actions();
            MatrixStats &stats = matrix_node->stats;

            if (stats.depth >= max_depth)
            {
                matrix_node->set_terminal();
                typename Types::ModelOutput model_output;
                model.inference(std::move(state), model_output);
                return {model_output.value.get_row_value(), model_output.value.get_row_value()};
            }

            const size_t rows = state.row_actions.size();
            const size_t cols = state.col_actions.size();
            matrix_node->expand(rows, cols);

            stats.chance_data_matrix.resize(0);
            stats.chance_data_matrix.fill(rows, cols);

            std::vector<int> &I = stats.I;
            std::vector<int> &J = stats.J;
            typename Types::VectorReal &row_solution = stats.row_solution;
            typename Types::VectorReal &col_solution = stats.col_solution;

            // I,J are the only places we can use prior solve info
            I.push_back(stats.row_pricipal_idx);
            J.push_back(stats.col_pricipal_idx);

            bool smaller_bounds = false;
            bool new_action = true;
            int latest_row_idx = stats.row_pricipal_idx;
            int latest_col_idx = stats.col_pricipal_idx;
            bool solved_exactly = true;

            while (!fuzzy_equals(alpha, beta) && (smaller_bounds || new_action))
            {

                // get entry values/bounds for newly expanded sub game, use last added actions

                for (const int row_idx : I)
                {
                    Data &data = stats.chance_data_matrix.get(row_idx, latest_col_idx);
                    ChanceNode *chance_node = matrix_node->access(row_idx, latest_col_idx);
                    solved_exactly &= try_solve_chance_node(max_depth, device, state, model, matrix_node, row_idx, latest_col_idx);
                }

                for (const int col_idx : J)
                {
                    Data &data = stats.chance_data_matrix.get(latest_row_idx, col_idx);
                    ChanceNode *chance_node = matrix_node->access(latest_row_idx, col_idx);
                    solved_exactly &= try_solve_chance_node(max_depth, device, state, model, matrix_node, latest_row_idx, col_idx);
                }

                // solve newly expanded and explored game

                int entry_idx = 0;
                if (solved_exactly)
                {
                    typename Types::MatrixValue matrix{I.size(), J.size()};
                    for (auto row_idx : I)
                    {
                        for (auto col_idx : J)
                        {
                            const Data &data = stats.chance_data_matrix.get(row_idx, col_idx);
                            matrix[entry_idx] = data.alpha_explored;
                            ++entry_idx;
                        }
                    }
                    LRSNash::solve(matrix, row_solution, col_solution);
                }
                else
                {
                    typename Types::MatrixValue alpha_matrix{I.size(), J.size()}, beta_matrix{I.size(), J.size()};
                    for (auto row_idx : I)
                    {
                        for (auto col_idx : J)
                        {
                            const Data &data = stats.chance_data_matrix.get(row_idx, col_idx);
                            alpha_matrix[entry_idx] = static_cast<Real>(data.alpha_explored + data.unexplored * min_val);
                            beta_matrix[entry_idx] = static_cast<Real>(data.beta_explored + data.unexplored * max_val);
                            ++entry_idx;
                        }
                    }
                    typename Types::VectorReal temp;
                    LRSNash::solve(alpha_matrix, row_solution, temp);
                    temp.clear();
                    LRSNash::solve(beta_matrix, temp, col_solution);
                }

                std::pair<int, Real>
                    iv = best_response_row(
                        max_depth,
                        device,
                        state,
                        model,
                        matrix_node,
                        alpha, max_val,
                        col_solution);

                std::pair<int, Real>
                    jv = best_response_col(
                        max_depth,
                        device,
                        state,
                        model,
                        matrix_node,
                        min_val, beta,
                        row_solution);

                // prune this node if no best response is as good as alpha/beta
                if (iv.first == -1)
                {
                    return {min_val, min_val};
                }
                if (jv.first == -1)
                {
                    return {max_val, max_val};
                }

                smaller_bounds = false;
                new_action = false;
                latest_row_idx = iv.first;
                latest_col_idx = jv.first;

                if (std::find(I.begin(), I.end(), latest_row_idx) == I.end())
                {
                    I.push_back(latest_row_idx);
                    new_action = true;
                }
                if (std::find(J.begin(), J.end(), latest_col_idx) == J.end())
                {
                    J.push_back(latest_col_idx);
                    new_action = true;
                }
                if (jv.second > alpha)
                {
                    alpha = jv.second;
                    smaller_bounds = true;
                }
                if (iv.second < beta)
                {
                    beta = iv.second;
                    smaller_bounds = true;
                }
            }

            stats.row_pricipal_idx = I[std::distance(row_solution.begin(), std::max_element(row_solution.begin(), row_solution.end()))];
            stats.col_pricipal_idx = J[std::distance(col_solution.begin(), std::max_element(col_solution.begin(), col_solution.end()))];

            // Now we correct row/col solution data in matrix node. Reorder, account for 'unsolved' i/j's, pad with zeros.
            typename Types::VectorReal temp_strategy{};
            temp_strategy.resize(rows);
            for (int i = 0; i < row_solution.size(); ++i)
            {
                temp_strategy[I[i]] = row_solution[i];
            }
            row_solution = temp_strategy;
            temp_strategy.clear();
            temp_strategy.resize(cols);
            for (int j = 0; j < col_solution.size(); ++j)
            {
                temp_strategy[J[j]] = col_solution[j];
            }
            col_solution = temp_strategy;

            math::canonicalize(alpha);
            math::canonicalize(beta);

            return {alpha, beta};
        }

        std::pair<int, Real>
        best_response_row(
            const size_t max_depth,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            const Real alpha, const Real beta,
            const Types::VectorReal &col_strategy) const
        {
            Real best_response{alpha};
            MatrixStats &stats = matrix_node->stats;
            std::vector<int> &I = stats.I;
            std::vector<int> &J = stats.J;
            int best_row_idx = -1;

            for (int row_idx = 0; row_idx < state.row_actions.size(); ++row_idx)
            {
                bool skip_exploration = (std::find(I.begin(), I.end(), row_idx) != I.end());
                const typename Types::Action row_action = state.row_actions[row_idx];

                Real max_priority{0}, expected_value{0}, total_unexplored{0};
                std::vector<Real> exploration_priorities;
                int col_idx, next_j;
                for (int j = 0; j < J.size(); ++j)
                {
                    const int col_idx_temp = J[j];
                    Data &data = stats.chance_data_matrix.get(row_idx, col_idx_temp);
                    // we still have to calculate expected score to return -1 if pruning is called for
                    expected_value += col_strategy[j] * data.beta_explored;

                    const Real priority =
                        skip_exploration
                            ? Real{0}
                            : Real{col_strategy[j] * data.unexplored};
                    total_unexplored += col_strategy[j] * data.unexplored;
                    exploration_priorities.push_back(priority);
                    if (priority > max_priority)
                    {
                        col_idx = col_idx_temp;
                        max_priority = priority;
                        next_j = j;
                    }
                }

                while (
                    (max_priority > Real{Rational<>{0}}) &&
                    (Real{expected_value + beta * total_unexplored} >= best_response))
                {
                    Data &data = stats.chance_data_matrix.get(row_idx, col_idx);
                    const typename Types::Action col_action = state.col_actions[col_idx];
                    if (data.chance_actions.size() == 0)
                    {
                        state.get_chance_actions(row_action, col_action, data.chance_actions);
                    }
                    if (data.next_chance_idx >= data.chance_actions.size())
                    {
                        break;
                    }
                    typename Types::State state_copy = state;
                    state_copy.apply_actions(row_action, col_action, data.chance_actions[data.next_chance_idx++]);
                    ChanceNode *chance_node = matrix_node->access(row_idx, col_idx);
                    MatrixNode *matrix_node_next = chance_node->access(state_copy.get_obs());
                    matrix_node_next->stats.depth = stats.depth + 1;
                    const typename Types::Prob prob = state_copy.prob;

                    auto alpha_beta_pair = double_oracle(
                        max_depth,
                        device,
                        state_copy,
                        model,
                        matrix_node_next,
                        min_val, max_val);

                    data.alpha_explored += alpha_beta_pair.first * prob;
                    data.beta_explored += alpha_beta_pair.second * prob;
                    expected_value += alpha_beta_pair.second * prob * col_strategy[next_j];

                    data.unexplored -= prob;
                    total_unexplored -= prob * col_strategy[next_j];
                    exploration_priorities[next_j] -= prob * col_strategy[next_j];

                    if constexpr (std::is_same_v<Real, mpq_class>)
                    {
                        assert(data.unexplored >= Real{0});
                        assert(total_unexplored >= Real{0});
                    }

                    max_priority = typename Types::Prob{typename Types::Q{0}};
                    for (int j = 0; j < J.size(); ++j)
                    {
                        const Real priority = exploration_priorities[j];
                        if (priority > max_priority)
                        {
                            col_idx = J[j];
                            max_priority = priority;
                            next_j = j;
                        }
                    }
                }

                expected_value += total_unexplored * beta;
                math::canonicalize(expected_value);

                if (expected_value >= best_response || (best_row_idx == -1 && fuzzy_equals(expected_value, best_response)))
                {
                    best_row_idx = row_idx;
                    best_response = expected_value;
                }
            }
            return {best_row_idx, best_response};
        }

        std::pair<int, Real> best_response_col(
            const size_t max_depth,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            const Real alpha, const Real beta,
            const Types::VectorReal &row_strategy) const
        {
            Real best_response{beta};
            MatrixStats &stats = matrix_node->stats;
            std::vector<int> &I = stats.I;
            std::vector<int> &J = stats.J;
            int best_col_idx = -1;

            for (int col_idx = 0; col_idx < state.col_actions.size(); ++col_idx)
            {
                bool skip_exploration = (std::find(J.begin(), J.end(), col_idx) != J.end());
                const typename Types::Action col_action = state.col_actions[col_idx];

                Real max_priority{0}, expected_value{0}, total_unexplored{0};
                std::vector<Real> exploration_priorities;
                int row_idx, next_i;
                for (int i = 0; i < I.size(); ++i)
                {
                    const int row_idx_temp = I[i];
                    Data &data = stats.chance_data_matrix.get(row_idx_temp, col_idx);
                    expected_value += row_strategy[i] * data.alpha_explored;

                    const Real priority =
                        skip_exploration
                            ? Real{0}
                            : Real{row_strategy[i] * data.unexplored};

                    total_unexplored += row_strategy[i] * data.unexplored;
                    exploration_priorities.push_back(priority);
                    if (priority > max_priority)
                    {
                        row_idx = row_idx_temp;
                        max_priority = priority;
                        next_i = i;
                    }
                }

                while (
                    fuzzy_greater(total_unexplored, Real{Rational<>{0}}) &&
                    (Real{expected_value + alpha * total_unexplored} <= best_response))
                {
                    Data &data = stats.chance_data_matrix.get(row_idx, col_idx);
                    const typename Types::Action row_action = state.row_actions[row_idx];
                    if (data.chance_actions.size() == 0)
                    {
                        state.get_chance_actions(row_action, col_action, data.chance_actions);
                    }

                    if (data.next_chance_idx >= data.chance_actions.size())
                    {
                        break;
                    }
                    typename Types::State state_copy = state;
                    state_copy.apply_actions(row_action, col_action, data.chance_actions[data.next_chance_idx++]);
                    ChanceNode *chance_node = matrix_node->access(row_idx, col_idx);
                    MatrixNode *matrix_node_next = chance_node->access(state_copy.get_obs());
                    matrix_node_next->stats.depth = stats.depth + 1;
                    const typename Types::Prob prob = state_copy.prob;

                    auto alpha_beta_pair = double_oracle(
                        max_depth,
                        device,
                        state_copy,
                        model,
                        matrix_node_next,
                        min_val, max_val);

                    data.alpha_explored += alpha_beta_pair.first * prob;
                    data.beta_explored += alpha_beta_pair.second * prob;
                    expected_value += alpha_beta_pair.first * prob * row_strategy[next_i];

                    data.unexplored -= prob;
                    total_unexplored -= prob * row_strategy[next_i];
                    exploration_priorities[next_i] -= prob * row_strategy[next_i];

                    if constexpr (std::is_same_v<Real, mpq_class>)
                    {
                        assert(data.unexplored >= Real{0});
                        assert(total_unexplored >= Real{0});
                    }

                    max_priority = Real{Rational<>{0}};
                    for (int i = 0; i < I.size(); ++i)
                    {
                        const Real priority = exploration_priorities[i];
                        if (priority > max_priority)
                        {
                            row_idx = I[i];
                            max_priority = priority;
                            next_i = i;
                        }
                    }
                }

                expected_value += total_unexplored * alpha;
                math::canonicalize(expected_value);

                if (expected_value <= best_response || (best_col_idx == -1 && fuzzy_equals(expected_value, best_response)))
                {
                    best_col_idx = col_idx;
                    best_response = expected_value;
                }
            }
            return {best_col_idx, best_response};
        }

    private:
        template <typename T>
        inline bool fuzzy_equals(T x, T y) const
        {
            if constexpr (std::is_same_v<T, mpq_class>)
            {
                math::canonicalize(x);
                math::canonicalize(y);
                return x == y;
            }
            else
            {
                static const Real epsilon{Rational{1, 1 << 24}};
                static const Real neg_epsilon{Rational{-1, 1 << 24}};
                T z{x - y};
                return neg_epsilon < z && z < epsilon;
            }
        }

        template <typename T>
        inline bool fuzzy_greater(T x, T y) const
        {
            if constexpr (std::is_same_v<T, mpq_class>)
            {
                return x > y;
            }
            else
            {
                static const Real epsilon{Rational{1, 1 << 24}};
                bool a = x > y + epsilon;
                return a;
            }
        }

        inline bool try_solve_chance_node(
            const size_t max_depth,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            int row_idx, int col_idx) const
        {
            MatrixStats &stats = matrix_node->stats;
            ChanceNode *chance_node = matrix_node->access(row_idx, col_idx);
            Data &data = stats.chance_data_matrix.get(row_idx, col_idx);

            if (data.unexplored > typename Types::Prob{0})
            {
                auto row_action = state.row_actions[row_idx];
                auto col_action = state.col_actions[col_idx];

                auto &chance_actions = data.chance_actions;
                if (chance_actions.size() == 0)
                {
                    state.get_chance_actions(row_action, col_action, chance_actions);
                }

                // go through all chance actions
                for (; data.next_chance_idx < chance_actions.size(); ++data.next_chance_idx)
                {
                    if (chance_actions.size() <= data.next_chance_idx)
                    {
                        break;
                    }

                    const typename Types::Obs chance_action = chance_actions[data.next_chance_idx];
                    typename Types::State state_copy = state;
                    state_copy.apply_actions(row_action, col_action, chance_action);
                    MatrixNode *matrix_node_next = chance_node->access(state_copy.get_obs());
                    matrix_node_next->stats.depth = stats.depth + 1;
                    const typename Types::Prob prob = state_copy.prob;

                    auto alpha_beta = double_oracle(max_depth, device, state_copy, model, matrix_node_next, min_val, max_val);

                    data.alpha_explored += alpha_beta.first * prob;
                    data.beta_explored += alpha_beta.second * prob;
                    data.unexplored -= prob;
                }
            }

            bool solved_exactly = (data.alpha_explored == data.beta_explored) && (data.unexplored == Real{Rational<>{0}});
            return solved_exactly;
        };

        Real row_alpha_beta(
            Types::State &state,
            typename Types::Model &model,
            MatrixNode *matrix_node,
            Real alpha,
            Real beta)
        {
            return max_val;
        }

        Real col_alpha_beta(
            Types::State &state,
            typename Types::Model &model,
            MatrixNode *matrix_node,
            Real alpha,
            Real beta)
        {
            return min_val;
        }
        // Serialized AlphaBeta - see paper
    };
};