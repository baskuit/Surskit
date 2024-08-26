#pragma once

#include <libpinyon/lrslib.h>
#include <types/matrix.h>
#include <algorithm/algorithm.h>
#include <tree/tree.h>

template <typename Types>
struct AlphaBetaForce : Types
{
    using Real = Types::Real;

    struct MatrixNode;
    struct Branch
    {
        Types::Prob prob;
        Types::Obs obs;
        Types::Seed seed;
        std::unique_ptr<MatrixNode> matrix_node;

        Branch(
            const Types::State &state,
            const Types::Seed &seed)
            : prob{state.prob},
              obs{state.get_obs()},
              seed{seed},
              matrix_node{std::make_unique<MatrixNode>()}
        {
        }
    };

    struct Data
    {
        Types::Prob unexplored{1};
        Real alpha_explored{0}, beta_explored{0};
        size_t tries{0};

        std::unordered_map<
            size_t,
            Branch>
            branches{};

        Data() {}
        Data(Data &&) {}

        friend std::ostream &operator<<(std::ostream &os, const Data &data)
        {
            if constexpr (std::is_same_v<typename Types::Real, mpq_class>)
            {
                os << '(' << data.alpha_explored.get_d() << " " << data.beta_explored.get_d() << " " << data.unexplored.get_d() << " " << data.tries << ")";
            }
            else
            {
                os << '(' << data.alpha_explored << " " << data.beta_explored << " " << data.unexplored << " " << data.tries << ")";
            }
            return os;
        }
    };

    struct MatrixNode
    {
        DataMatrix<Data> chance_data_matrix{};
        Types::VectorReal row_solution{}, col_solution{};
        unsigned int depth = 0;

        int row_pricipal_idx = 0, col_pricipal_idx = 0;
        std::vector<int> I{}, J{};

        Real alpha, beta;

        size_t count_matrix_nodes() const
        {
            size_t c = 1;
            for (const auto &data : chance_data_matrix)
            {
                for (const auto &pair : data.branches)
                {
                    const auto &branch = pair.second;
                    c += branch.matrix_node->count_matrix_nodes();
                }
            }
            return c;
        }
    };

    class Search
    {
    public:
        const Real min_val{0};
        const Real max_val{1};

        const size_t min_tries{0};
        const size_t max_tries{1 << 6};
        const Types::Prob max_unexplored{typename Types::Q{0}};
        const Types::ObsHash hash_function{};

        Search() {}

        Search(size_t max_tries) : max_tries{max_tries} {}

        Search(size_t min_tries, size_t max_tries, Types::Prob max_unexplored)
            : min_tries{min_tries}, max_tries{max_tries}, max_unexplored{max_unexplored} {}

        Search(
            Real min_val, Real max_val,
            size_t min_tries, size_t max_tries, Types::Prob max_unexplored)
            : min_val(min_val), max_val(max_val),
              min_tries{min_tries}, max_tries{max_tries}, max_unexplored{max_unexplored} {}

        size_t run(
            const size_t max_depth,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode &root) const
        {
            auto start = std::chrono::high_resolution_clock::now();
            typename Types::State state_copy{state};
            double_oracle(max_depth, device, state_copy, model, &root, min_val, max_val);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            return duration;
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
                const typename Types::Value payoff = state.get_payoff();
                matrix_node->alpha = payoff.get_row_value();
                matrix_node->beta = payoff.get_row_value();
                return {payoff.get_row_value(), payoff.get_row_value()};
            }

            state.get_actions(); // here for monte carlo model

            if (matrix_node->depth >= max_depth)
            {
                typename Types::ModelOutput model_output;
                model.inference(std::move(state), model_output);
                matrix_node->alpha = model_output.value.get_row_value();
                matrix_node->beta = model_output.value.get_row_value();
                return {model_output.value.get_row_value(), model_output.value.get_row_value()};
            }

            const size_t rows = state.row_actions.size();
            const size_t cols = state.col_actions.size();

            DataMatrix<Data> &data_matrix = matrix_node->chance_data_matrix;
            if (data_matrix.size() == 0)
            {
                data_matrix.rows = rows;
                data_matrix.cols = cols;
                for (int i = 0; i < rows * cols; ++i)
                {
                    data_matrix.emplace_back();
                }
            } // No else here, not intended to iterate

            std::vector<int> &I = matrix_node->I;
            std::vector<int> &J = matrix_node->J;
            I.clear();
            J.clear();
            // alpha = min_val;
            // beta = max_val;alpha
            // current best strategy. used to calculate best response values
            // entries correspond to I, J entires and order. It is a permuted submatrix of the full matrix, basically.
            typename Types::VectorReal &row_solution = matrix_node->row_solution;
            typename Types::VectorReal &col_solution = matrix_node->col_solution;

            // I,J are the only places we can use prior solve info
            // either already there or init to 0
            I.push_back(matrix_node->row_pricipal_idx);
            J.push_back(matrix_node->col_pricipal_idx);

            bool smaller_bounds = false;
            bool new_action = true;
            int latest_row_idx = matrix_node->row_pricipal_idx;
            int latest_col_idx = matrix_node->col_pricipal_idx;
            bool solved_exactly = true;

            while (!fuzzy_equals(alpha, beta) && (smaller_bounds || new_action))
            {

                // get entry values/bounds for newly expanded sub game, use last added actions

                for (const int row_idx : I)
                {
                    Data &data = matrix_node->chance_data_matrix.get(row_idx, latest_col_idx);
                    solved_exactly &= try_solve_chance_branches(max_depth, device, state, model, matrix_node, row_idx, latest_col_idx);
                }

                for (const int col_idx : J)
                {
                    Data &data = matrix_node->chance_data_matrix.get(latest_row_idx, col_idx);
                    solved_exactly &= try_solve_chance_branches(max_depth, device, state, model, matrix_node, latest_row_idx, col_idx);
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
                            const Data &data = matrix_node->chance_data_matrix.get(row_idx, col_idx);
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
                            const Data &data = matrix_node->chance_data_matrix.get(row_idx, col_idx);
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

            matrix_node->row_pricipal_idx = I[std::distance(row_solution.begin(), std::max_element(row_solution.begin(), row_solution.end()))];
            matrix_node->col_pricipal_idx = J[std::distance(col_solution.begin(), std::max_element(col_solution.begin(), col_solution.end()))];

            // Now we correct row/col solution data in matrix node. Reorder, account for 'unsolved' i/j's, pad with zeros.
            typename Types::VectorReal temp_strategy{};
            temp_strategy.resize(rows, Real{0});
            for (int i = 0; i < row_solution.size(); ++i)
            {
                temp_strategy[I[i]] = row_solution[i];
            }
            row_solution = temp_strategy;
            temp_strategy.clear();
            temp_strategy.resize(cols, Real{0});
            for (int j = 0; j < col_solution.size(); ++j)
            {
                temp_strategy[J[j]] = col_solution[j];
            }
            col_solution = temp_strategy;

            math::canonicalize(alpha);
            math::canonicalize(beta);
            matrix_node->alpha = alpha;
            matrix_node->beta = beta;
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
            std::vector<int> &I = matrix_node->I;
            std::vector<int> &J = matrix_node->J;
            int best_row_idx = -1;

            for (int row_idx = 0; row_idx < state.row_actions.size(); ++row_idx)
            {
                const auto row_action = state.row_actions[row_idx];

                const bool skip_exploration = (std::find(I.begin(), I.end(), row_idx) != I.end());

                // 'priority' is the product of the opposing and 'chance' players probability
                // its clearly what we want to minimize rather than just the chance component
                Real max_priority{0}, expected_value{0}, total_unexplored{0};
                std::vector<Real> exploration_priorities{};
                int col_idx, next_j;
                for (int j = 0; j < J.size(); ++j)
                {
                    const int col_idx_temp = J[j];
                    Data &data = matrix_node->chance_data_matrix.get(row_idx, col_idx_temp);
                    // we still have to calculate expected score to return -1 if pruning is called for
                    expected_value += col_strategy[j] * data.beta_explored;

                    const Real priority =
                        (skip_exploration || (data.tries >= max_tries))
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
                    (max_priority > Real{0}) &&
                    (Real{expected_value + beta * total_unexplored} >= best_response))
                {
                    const auto col_action = state.col_actions[col_idx];

                    Data &data = matrix_node->chance_data_matrix.get(row_idx, col_idx);

                    bool produced_new_branch = false;
                    for (; (data.tries < max_tries) &&
                           ((data.tries < min_tries) || (data.unexplored > max_unexplored));
                         ++data.tries)
                    {
                        typename Types::State state_copy{state};
                        const typename Types::Seed seed{device.uniform_64()};
                        state_copy.randomize_transition(seed);
                        state_copy.apply_actions(row_action, col_action);
                        const size_t obs_hash = hash_function(state_copy.get_obs());

                        if (data.branches.find(obs_hash) == data.branches.end())
                        {
                            produced_new_branch = true;

                            data.branches.try_emplace(
                                obs_hash, state_copy, seed);
                            Branch &new_branch = data.branches.at(obs_hash);

                            new_branch.matrix_node->depth = matrix_node->depth + 1;
                            const auto prob = new_branch.prob;

                            auto alpha_beta_pair = double_oracle(
                                max_depth,
                                device,
                                state_copy,
                                model,
                                new_branch.matrix_node.get(),
                                min_val, max_val);

                            data.alpha_explored += alpha_beta_pair.first * prob;
                            data.beta_explored += alpha_beta_pair.second * prob;
                            expected_value += alpha_beta_pair.second * prob * col_strategy[next_j];

                            data.unexplored -= prob;
                            total_unexplored -= prob * col_strategy[next_j];
                            exploration_priorities[next_j] -= prob * col_strategy[next_j];

                            break;
                        }
                    }

                    if (!produced_new_branch)
                    {
                        exploration_priorities[next_j] = typename Types::Prob{typename Types::Q{0}};
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

        std::pair<int, Real>
        best_response_col(
            const size_t max_depth,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            const Real alpha, const Real beta,
            const Types::VectorReal &row_strategy) const
        {
            Real best_response{beta};
            std::vector<int> &I = matrix_node->I;
            std::vector<int> &J = matrix_node->J;
            int best_col_idx = -1;

            for (int col_idx = 0; col_idx < state.col_actions.size(); ++col_idx)
            {
                const auto col_action = state.col_actions[col_idx];

                const bool skip_exploration = (std::find(J.begin(), J.end(), col_idx) != J.end());

                // 'priority' is the product of the opposing and 'chance' players probability
                // its clearly what we want to minimize rather than just the chance component
                Real max_priority{0}, expected_value{0}, total_unexplored{0};
                std::vector<Real> exploration_priorities{};
                int row_idx, next_i;
                for (int i = 0; i < I.size(); ++i)
                {
                    const int row_idx_temp = I[i];
                    Data &data = matrix_node->chance_data_matrix.get(row_idx_temp, col_idx);
                    // we still have to calculate expected score to return -1 if pruning is called for
                    expected_value += row_strategy[i] * data.alpha_explored;

                    const Real priority =
                        (skip_exploration || (data.tries >= max_tries))
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
                    (max_priority > Real{0}) &&
                    (Real{expected_value + alpha * total_unexplored} <= beta))
                {
                    const auto row_action = state.row_actions[row_idx];

                    Data &data = matrix_node->chance_data_matrix.get(row_idx, col_idx);

                    bool produced_new_branch = false;
                    for (; (data.tries < max_tries) &&
                           ((data.tries < min_tries) || (data.unexplored > max_unexplored));
                         ++data.tries)
                    {
                        typename Types::State state_copy{state};
                        const typename Types::Seed seed{device.uniform_64()};
                        state_copy.randomize_transition(seed);
                        state_copy.apply_actions(row_action, col_action);
                        // const size_t obs_hash = state_copy.get_obs().get();
                        const size_t obs_hash = hash_function(state_copy.get_obs());

                        if (data.branches.find(obs_hash) == data.branches.end())
                        {
                            produced_new_branch = true;

                            data.branches.try_emplace(
                                obs_hash, state_copy, seed);
                            Branch &new_branch = data.branches.at(obs_hash);

                            new_branch.matrix_node->depth = matrix_node->depth + 1;
                            const auto prob = new_branch.prob;

                            auto alpha_beta_pair = double_oracle(
                                max_depth,
                                device,
                                state_copy,
                                model,
                                new_branch.matrix_node.get(),
                                min_val, max_val); // TODO alpha/beta

                            data.alpha_explored += alpha_beta_pair.first * prob;
                            data.beta_explored += alpha_beta_pair.second * prob;
                            expected_value += alpha_beta_pair.first * prob * row_strategy[next_i];

                            data.unexplored -= prob;
                            total_unexplored -= prob * row_strategy[next_i];
                            exploration_priorities[next_i] -= prob * row_strategy[next_i];

                            break;
                        }
                    }

                    if (!produced_new_branch)
                    {
                        exploration_priorities[next_i] = typename Types::Prob{typename Types::Q{0}};
                    }

                    max_priority = typename Types::Prob{typename Types::Q{0}};
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

        template <template <typename> typename Wrapper, typename T>
        inline bool fuzzy_greater(Wrapper<T> x, Wrapper<T> y) const
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

        inline bool try_solve_chance_branches(
            const size_t max_depth,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            int row_idx, int col_idx) const
        {
            Data &data = matrix_node->chance_data_matrix.get(row_idx, col_idx);

            const auto row_action = state.row_actions[row_idx];
            const auto col_action = state.col_actions[col_idx];

            for (; (data.tries < max_tries) && (data.unexplored > typename Types::Prob{0}) &&
                   ((data.tries < min_tries) || (data.unexplored > max_unexplored));
                 ++data.tries)
            {
                typename Types::State state_copy{state};
                const typename Types::Seed seed{device.uniform_64()};
                state_copy.randomize_transition(seed);
                state_copy.apply_actions(row_action, col_action);
                // const size_t obs_hash = state_copy.get_obs().get();
                const size_t obs_hash = hash_function(state_copy.get_obs());

                if (data.branches.find(obs_hash) == data.branches.end())
                {
                    data.branches.try_emplace(
                        obs_hash, state_copy, seed);
                    Branch &new_branch = data.branches.at(obs_hash);

                    new_branch.matrix_node->depth = matrix_node->depth + 1;

                    const auto alpha_beta =
                        double_oracle(
                            max_depth,
                            device,
                            state_copy,
                            model,
                            new_branch.matrix_node.get(),
                            min_val, max_val);

                    data.alpha_explored += alpha_beta.first * new_branch.prob;
                    data.beta_explored += alpha_beta.second * new_branch.prob;
                    data.unexplored -= new_branch.prob;
                }
            }

            const bool solved_exactly = (data.alpha_explored == data.beta_explored) && (data.unexplored == Real{Rational<>{0}});
            return solved_exactly;
        };

        Real row_alpha_beta(
            Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            Real alpha,
            Real beta)
        {
            return max_val;
        }

        Real col_alpha_beta(
            Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            Real alpha,
            Real beta)
        {
            return min_val;
        }
        // Serialized AlphaBeta, TODO
    };
};
