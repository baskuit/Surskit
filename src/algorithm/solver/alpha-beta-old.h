#pragma once

#include <tree/tree.h>
#include <algorithm/algorithm.h>

/*

An implementation of Simultaneous Move Alpha Beta

See "Using Double-Oracle Method and Serialized Alpha-Beta Search
for Pruning in Simultaneous Move Games>

Note: This utility is deprecated in favor of `alpha-beta.h`, which allows for

    alpha < beta

However, if that is not a possibility then this implementation should be faster.

Note: also is modified for stochastic SM games (it requires get_chance_actions().)

and it also has an improvement on the bounds for skipping and action while checking the best response.

*/

template <
    IsSingleModelTypes Types,
    template <typename...> typename NodePair = DefaultNodes>
struct AlphaBetaOld : Types
{
    // for appearances
    using Real = typename Types::Real;

    struct MatrixStats
    {
        // value for the maximizing/row player.
        Real row_value;

        // matrices of pessimistic/optimisitic values. always over full actions
        Types::MatrixReal p;
        Types::MatrixReal o;

        // vector of row_idx, col_idx in the substage
        std::vector<int> I{}, J{};

        Types::VectorReal row_solution, col_solution;
        int row_br_idx, col_br_idx;
        Types::MatrixInt chance_actions_solved;

        size_t matrix_node_count = 1;
        size_t matrix_node_count_last = 0;

        unsigned int depth = 0;
        Types::Prob prob;
    };
    struct ChanceStats
    {
        typename Types::Prob explored{0};
        size_t matrix_node_count = 0;
    };
    using MatrixNode = NodePair<Types, MatrixStats, ChanceStats>::MatrixNode;
    using ChanceNode = NodePair<Types, MatrixStats, ChanceStats>::ChanceNode;

    class Search
    {
    public:
        const Real min_val{0};
        const Real max_val{1};

        int max_depth = -1;
        Real epsilon{Rational{1, 1 << 24}};

        Search() {}
        Search(Real min_val, Real max_val) : min_val(min_val), max_val(max_val) {}

        void run(
            Types::State &state,
            Types::Model &model,
            MatrixNode *root)
        {
            double_oracle(state, model, root, min_val, max_val);
        }

        Real double_oracle(
            Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            Real alpha,
            Real beta)
        {
            auto &stats = matrix_node->stats;
            state.get_actions();
            matrix_node->expand(state);

            if (state.is_terminal())
            {
                matrix_node->set_terminal();
                stats.row_value = state.get_payoff().get_row_value();
                return stats.row_value;
            }
            if (max_depth > 0 && stats.depth >= max_depth)
            {
                matrix_node->set_terminal();
                typename Types::ModelOutput model_output;
                model.inference(state, model_output);
                stats.row_value = model_output.value.get_row_value();
                return stats.row_value;
            }

            // 6: initialize restricted action sets I and J with a first action in stage s
            std::vector<int> &I = stats.I;
            std::vector<int> &J = stats.J;
            // TODO first action using info from last expansion, if possible
            I.push_back(0);
            J.push_back(0); // we assume index 0 in the actions is 'principal'

            typename Types::MatrixReal &p = stats.p;
            typename Types::MatrixReal &o = stats.o;
            // 7: pI,J ← alpha-betaMin (sI,J , minval, maxval)
            p.fill(state.row_actions.size(), state.col_actions.size(), min_val);
            // 8: oI,J ← alpha-betaMax (sI,J , minval, maxval)
            o.fill(state.row_actions.size(), state.col_actions.size(), max_val);
            stats.chance_actions_solved.fill(state.row_actions.size(), state.col_actions.size(), 0);
            // Note: this implementation does not use serialized alpha beta
            // Just seems like too much tree traversal?
            // 9: repeat, 23: until α = β

            while (!fuzzy_equals(alpha, beta))
            {
                // 10: for i ∈ I, j ∈ J do
                for (int row_idx : stats.I)
                {
                    for (int col_idx : stats.J)
                    {

                        Real &p_ij = p.get(row_idx, col_idx);
                        Real &o_ij = o.get(row_idx, col_idx);
                        // 11: if pi, j < oi, j then
                        if (p_ij < o_ij)
                        {
                            // 12: u(si, j ) ← double-oracle(si, j , pi, j, oi, j )
                            Real u_ij = typename Types::Q{0};
                            ChanceNode *chance_node = matrix_node->access(row_idx, col_idx);

                            const typename Types::Action row_action = state.row_actions[row_idx];
                            const typename Types::Action col_action = state.col_actions[col_idx];

                            std::vector<typename Types::Obs> chance_actions;
                            state.get_chance_actions(row_action, col_action, chance_actions);
                            for (const auto chance_action : chance_actions)
                            {

                                auto state_copy = state;
                                state_copy.apply_actions(row_action, col_action, chance_action);
                                MatrixNode *matrix_node_next = chance_node->access(state_copy.get_obs());

                                u_ij += double_oracle(state_copy, model, matrix_node_next, p_ij, o_ij) * state_copy.prob;
                                // u_ij is the value of chance node, i.e. expected score over all child matrix nodes

                                chance_node->stats.matrix_node_count += matrix_node_next->stats.matrix_node_count - matrix_node_next->stats.matrix_node_count_last;
                                matrix_node_next->stats.matrix_node_count_last = matrix_node_next->stats.matrix_node_count;

                                stats.chance_actions_solved.get(row_idx, col_idx) += 1;
                                matrix_node_next->stats.depth = stats.depth + 1;
                                matrix_node_next->stats.prob = state.prob;
                            }

                            // 13: pi, j ← u(si, j ); oi, j ← u(si, j )
                            p_ij = u_ij;
                            o_ij = u_ij;
                        }
                    }
                }

                // 14: <u(s), (x,y)> ← ComputeNE(I, J)
                typename Types::MatrixValue matrix;
                typename Types::VectorReal row_strategy, col_strategy;
                stats.row_value = solve_submatrix(matrix, matrix_node, row_strategy, col_strategy);
                // 15: hi0, vMaxi ← BRMax (s, α, y)
                auto iv = best_response_row(state, model, matrix_node, alpha, col_strategy);
                // 16: h j0 , vMini ← BRMin(s, β, x)
                auto jv = best_response_col(state, model, matrix_node, beta, row_strategy);
                stats.row_br_idx = iv.first;
                stats.col_br_idx = jv.first;
                stats.row_solution = row_strategy;
                stats.col_solution = col_strategy;

                // 17 - 20
                if (iv.first == -1)
                {
                    stats.row_value = min_val;
                    return min_val;
                }
                if (jv.first == -1)
                {
                    stats.row_value = max_val;
                    return max_val;
                }
                // 21: α ← max(α, vMin); β ← min(β, v Max )
                alpha = std::max(alpha, jv.second);
                beta = std::min(beta, iv.second);

                // 22: I ← I ∪ {i0}; J ← J ∪ { j0 }
                if (std::find(I.begin(), I.end(), iv.first) == I.end())
                {
                    I.push_back(iv.first);
                }
                if (std::find(J.begin(), J.end(), jv.first) == J.end())
                {
                    J.push_back(jv.first);
                }
                stats.row_value = alpha;
            }

            auto chance_node = matrix_node->child;
            while (chance_node != nullptr)
            {
                stats.matrix_node_count += chance_node->stats.matrix_node_count;
                chance_node = chance_node->next;
            }

            return stats.row_value;
        }

        std::pair<int, Real> 
        best_response_row(
            Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            Real alpha,
            Types::VectorReal &col_strategy)
        {
            typename Types::MatrixReal &p = matrix_node->stats.p;
            typename Types::MatrixReal &o = matrix_node->stats.o;
            std::vector<int> &I = matrix_node->stats.I;
            std::vector<int> &J = matrix_node->stats.J;

            // std::vector<int> sorted_col_idx{};
            // int max_idx = 0;
            // std::sort(J.begin(), J.end(), [&max_idx, &col_strategy](int col_idx_1, int col_idx_2)
            //           {
            //     // TODO set max_idx to the highest index that has non-zero
            //     return col_strategy[col_idx_1] < col_strategy[col_idx_2]; });

            // 1: BRvalue ← α
            Real best_response_row = alpha;
            // 2: i BR ← null
            int new_action_idx = -1;

            // 3: for i = {1, . . . , n} do
            for (int row_idx = 0; row_idx < state.row_actions.size(); ++row_idx)
            {
                bool skip_row_idx = false;
                // 4: pi,J ← alpha-betaMin(si,J , minval, maxval)
                // 5: oi,J ← alpha-betaMax (si,J , minval, maxval)
                // We don't perform serialized alpha-beta
                // Furthermore, I'm not sure what this step does that lines 7, 8 in double_oracle don't

                Real expected_o_payoff = typename Types::Q{0};
                for (int j = 0; j < J.size(); ++j)
                {
                    expected_o_payoff += col_strategy[j] * o.get(row_idx, J[j]);
                }

                // 6: for j ∈ J; y j > 0 ∧ pi, j < oi, j do
                for (int j = 0; j < J.size(); ++j)
                {
                    const int col_idx = J[j];
                    Real y = col_strategy[j];

                    Real &p_ij = p.get(row_idx, col_idx);
                    Real &o_ij = o.get(row_idx, col_idx);
                    if (y > Real{0} && p_ij < o_ij)
                    {

                        // we are iterating over joint actions/entries here that remain unsolved

                        // 7: p0 i, j ← max pi, j , BRvalue − P j0 ∈Jr{ j} y j0 · oi, j0
                        const Real p__ij = std::max(p_ij.value, ((best_response_row - expected_o_payoff + y * o_ij)).value);
                        const Real p__ij_ = std::max(p_ij.value, ((best_response_row - expected_o_payoff + y * o_ij) / y).value);

                        bool skip = p__ij > o_ij;
                        bool skip_ = p__ij_ > o_ij;
                        // 8: if p0 > oi, j then
                        if (skip_)
                        {
                            skip_row_idx = true;
                            break;
                        }
                        else
                        {
                            // 11: u(s_ij) = double_oracle (s_ij, p_ij, o_ij)
                            Real u_ij = typename Types::Q{0};
                            ChanceNode *chance_node = matrix_node->access(row_idx, col_idx);

                            const typename Types::Action row_action = state.row_actions[row_idx];
                            const typename Types::Action col_action = state.col_actions[col_idx];

                            std::vector<typename Types::Obs> chance_actions;
                            state.get_chance_actions(row_action, col_action, chance_actions);
                            for (const auto chance_action : chance_actions)
                            {
                                auto state_copy = state;
                                state_copy.apply_actions(row_action, col_action, chance_action);
                                MatrixNode *matrix_node_next = chance_node->access(state_copy.get_obs());
                                u_ij += double_oracle(state_copy, model, matrix_node_next, p_ij, o_ij) * state_copy.prob;
                                chance_node->stats.explored += state_copy.prob;

                                chance_node->stats.matrix_node_count += matrix_node_next->stats.matrix_node_count - matrix_node_next->stats.matrix_node_count_last;
                                matrix_node_next->stats.matrix_node_count_last = matrix_node_next->stats.matrix_node_count;

                                // auto best_possible = (chance_node->stats.explored * Real{-1} + Real{1}) + u_ij;
                                // if (best_possible < p__ij && chance_node->stats.explored < Real{1})
                                // {
                                //     skip_row_idx = true;
                                //     // break;
                                // }
                            }

                            // 12: pi, j ← u(si, j ); oi, j ← u(si, j)
                            p_ij = u_ij;
                            o_ij = u_ij;

                            // idk what this does, but not removing TODO
                            // if (!skip_row_idx)
                            // {
                            //     p_ij = u_ij;
                            //     o_ij = u_ij;
                            // }
                            // else
                            // {
                            //     p_ij = u_ij;
                            //     o_ij = (chance_node->stats.explored * Real{-1} + Real{1}) + u_ij;
                            // }
                        }
                    }
                }

                // 9: continue with next i
                if (skip_row_idx)
                {
                    continue;
                }

                // 13 - 14
                Real expected_row_payoff = typename Types::Q{0};
                for (int j = 0; j < J.size(); ++j)
                {
                    const int col_idx = J[j];
                    expected_row_payoff += col_strategy[j] * o.get(row_idx, col_idx);
                } // o_ij = u_ij by now
                if (expected_row_payoff >= best_response_row || (new_action_idx == -1 && fuzzy_equals(expected_row_payoff, best_response_row)))
                {
                    new_action_idx = row_idx;
                    best_response_row = expected_row_payoff;
                }
            } // end row action loop

            std::pair<int, Real> pair{new_action_idx, best_response_row};
            return pair;
        }

        std::pair<int, Real> 
        best_response_col(
            Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            Real beta,
            Types::VectorReal &row_strategy)
        {
            typename Types::MatrixReal &p = matrix_node->stats.p;
            typename Types::MatrixReal &o = matrix_node->stats.o;
            std::vector<int> &I = matrix_node->stats.I;
            std::vector<int> &J = matrix_node->stats.J;

            Real best_response_col = beta;
            int new_action_idx = -1;

            for (int col_idx = 0; col_idx < state.col_actions.size(); ++col_idx)
            {
                bool cont = false;

                Real expected_p_payoff = typename Types::Q{0};
                for (int i = 0; i < I.size(); ++i)
                {
                    expected_p_payoff += row_strategy[i] * p.get(I[i], col_idx);
                }

                for (int i = 0; i < I.size(); ++i)
                {
                    const int row_idx = I[i];
                    Real x = row_strategy[i];

                    Real &p_ij = p.get(row_idx, col_idx);
                    Real &o_ij = o.get(row_idx, col_idx);

                    const bool ok = p_ij < o_ij;
                    if (x > Real{0} && ok)
                    {
                        const Real o__ij = std::min((o_ij).value, ((best_response_col - expected_p_payoff + x * p_ij)).value);
                        const Real o__ij_ = std::min((o_ij).value, ((best_response_col - expected_p_payoff + x * p_ij) / x).value);
                        bool skip = o__ij < p_ij;
                        bool skip_ = o__ij_ < p_ij;

                        if (skip_)
                        {
                            cont = true;
                            break;
                        }
                        else
                        {
                            Real u_ij = typename Types::Q{0};
                            ChanceNode *chance_node = matrix_node->access(row_idx, col_idx);

                            const typename Types::Action row_action = state.row_actions[row_idx];
                            const typename Types::Action col_action = state.col_actions[col_idx];

                            std::vector<typename Types::Obs> chance_actions;
                            state.get_chance_actions(row_action, col_action, chance_actions);
                            for (const auto chance_action : chance_actions)
                            {
                                auto state_copy = state;
                                state_copy.apply_actions(row_action, col_action, chance_action);
                                MatrixNode *matrix_node_next = chance_node->access(state_copy.get_obs());
                                u_ij += double_oracle(state_copy, model, matrix_node_next, p_ij, o_ij) * state_copy.prob;
                                chance_node->stats.explored += state_copy.prob;

                                chance_node->stats.matrix_node_count += matrix_node_next->stats.matrix_node_count - matrix_node_next->stats.matrix_node_count_last;
                                matrix_node_next->stats.matrix_node_count_last = matrix_node_next->stats.matrix_node_count;
                            }

                            // 12: pi, j ← u(si, j ); oi, j ← u(si, j)
                            p_ij = u_ij;
                            o_ij = u_ij;
                        }
                    }
                }

                if (cont)
                {
                    continue;
                }

                Real expected_col_payoff = typename Types::Q{0};
                for (int i = 0; i < I.size(); ++i)
                {
                    expected_col_payoff += row_strategy[i] * p.get(I[i], col_idx);
                }
                if (expected_col_payoff <= best_response_col || (new_action_idx == -1 && fuzzy_equals(expected_col_payoff, best_response_col)))
                {
                    new_action_idx = col_idx;
                    best_response_col = expected_col_payoff;
                }
            }

            std::pair<int, Real> pair{new_action_idx, best_response_col};
            return pair;
        }

    private:
        template <template <typename> typename Wrapper, typename T>
        bool fuzzy_equals(Wrapper<T> x, Wrapper<T> y)
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
                Wrapper<T> z{x - y};
                return neg_epsilon < z && z < epsilon;
            }
        }

        Real solve_submatrix(
            Types::MatrixValue &submatrix,
            MatrixNode *matrix_node,
            Types::VectorReal &row_strategy,
            Types::VectorReal &col_strategy)
        {
            // define submatrix
            submatrix.fill(matrix_node->stats.I.size(), matrix_node->stats.J.size());
            row_strategy.resize(submatrix.rows);
            col_strategy.resize(submatrix.cols);
            int entry_idx = 0;
            for (const int row_idx : matrix_node->stats.I)
            {
                for (const int col_idx : matrix_node->stats.J)
                {
                    submatrix[entry_idx++] = typename Types::Value{matrix_node->stats.p.get(row_idx, col_idx)};
                    // assert(matrix_node->stats.p.get(row_idx, col_idx) == matrix_node->stats.o.get(row_idx, col_idx));
                    // we can use either p or q here since the substage is solved
                }
            }

            LRSNash::solve(submatrix, row_strategy, col_strategy);

            Real value = typename Types::Q{0};
            for (int row_idx = 0; row_idx < submatrix.rows; ++row_idx)
            {
                for (int col_idx = 0; col_idx < submatrix.cols; ++col_idx)
                {
                    value += submatrix.get(row_idx, col_idx).get_row_value() * row_strategy[row_idx] * col_strategy[col_idx];
                }
            }
            return value;
        }

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
    };
    // Serialized AlphaBetaOld, TODO
};