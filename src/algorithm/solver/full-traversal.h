#pragma once

#include <libpinyon/lrslib.h>
#include <model/model.h>
#include <tree/tree.h>
#include <algorithm/algorithm.h>

#include <string>
#include <cassert>
#include <thread>

/*
    This algorithm expands a node into a tree that is one-to-one with the abstract game tree
    and iteratively solves it.

    The tree that is generated by this algorithm can then be wrapped in a TraversedState,
    which is a way of turning any StateChance into a SolvedState
*/

template <
    IsSingleModelTypes Types,
    template <typename...> typename NodePair = DefaultNodes>
    requires IsChanceStateTypes<Types>
struct FullTraversal : Types
{
    struct MatrixStats
    {
        Types::Value payoff{};
         Types::VectorReal row_solution, col_solution;
         Types::MatrixValue nash_payoff_matrix;

        size_t matrix_node_count = 1;
        unsigned int depth{};
         Types::Prob prob;

         Types::Mutex mutex{};
        bool is_expanded = false;
    };
    struct ChanceStats
    {
        std::vector<typename Types::Obs> chance_actions;
        std::vector<typename Types::Prob> chance_strategy;
        Types::Mutex mutex{};
        bool is_solved{false};
    };
    using MatrixNode = typename NodePair<Types, MatrixStats, ChanceStats>::MatrixNode;
    using ChanceNode = typename NodePair<Types, MatrixStats, ChanceStats>::ChanceNode;

    class Search
    {
    public:
        Search() {}

        std::pair<typename Types::Real, typename Types::Real>
        run(
            const size_t max_depth,
            Types::PRNG &,
            const Types::State &state,
            Types::Model &model,
            MatrixNode &matrix_node,
            const size_t threads = 1) const
        {
            auto state_ = state;
            std::thread thread_pool[threads];
            for (int i = 0; i < threads; ++i)
            {
                thread_pool[i] = std::thread(
                    &Search::run_, this, max_depth, &state_, &model, &matrix_node);
            }
            for (int i = 0; i < threads; ++i)
            {
                thread_pool[i].join();
            }
            return {matrix_node.stats.payoff.get_row_value(), matrix_node.stats.payoff.get_row_value()};
        }

        // use device to simplify SearchModel support, but not needed
        std::pair<typename Types::Real, typename Types::Real>
        run(
            const size_t max_depth,
            const Types::State &state,
            Types::Model &model,
            MatrixNode &matrix_node) const
        {
            auto state_{state};
            run_(max_depth, &state_, &model, &matrix_node);
            return {matrix_node.stats.payoff.get_row_value(), matrix_node.stats.payoff.get_row_value()};
        }

        void run_(
            const size_t max_depth,
            Types::State *state_ptr,
            Types::Model *model_ptr,
            MatrixNode *matrix_node) const
        {
            typename Types::State &state = *state_ptr;
            typename Types::Model &model = *model_ptr;
            // expand node
            state.get_actions();
            const size_t rows = state.row_actions.size();
            const size_t cols = state.col_actions.size();
            matrix_node->expand(rows, cols);

            MatrixStats &stats = matrix_node->stats;

            if (state.is_terminal())
            {
                stats.payoff = state.get_payoff();
                matrix_node->set_terminal();
                return;
            }
            if (stats.depth >= max_depth)
            {
                typename Types::ModelOutput output;
                model.inference(std::move(state), output);
                stats.payoff = output.value;
                matrix_node->set_terminal();
                return;
            }

            stats.mutex.lock();
            if (!stats.is_expanded)
            {
                stats.nash_payoff_matrix.fill(rows, cols);
                stats.row_solution.resize(rows);
                stats.col_solution.resize(cols);
                for (int row_idx = 0; row_idx < rows; ++row_idx)
                {
                    for (int col_idx = 0; col_idx < cols; ++col_idx)
                    {
                        const typename Types::Action &row_action{state.row_actions[row_idx]};
                        const typename Types::Action &col_action{state.col_actions[col_idx]};

                        ChanceNode *chance_node = matrix_node->access(row_idx, col_idx);
                    }
                }
            }
            stats.is_expanded = true;
            stats.mutex.unlock();

            // recurse
            for (int row_idx = 0; row_idx < rows; ++row_idx)
            {
                for (int col_idx = 0; col_idx < cols; ++col_idx)
                {
                    const typename Types::Action &row_action{state.row_actions[row_idx]};
                    const typename Types::Action &col_action{state.col_actions[col_idx]};

                    ChanceNode *chance_node = matrix_node->access(row_idx, col_idx);

                    if (!chance_node->stats.mutex.try_lock())
                    {
                        continue;
                    }

                    if (chance_node->stats.is_solved)
                    {
                        continue;
                    }

                    auto &chance_actions = chance_node->stats.chance_actions;
                    state.get_chance_actions(row_action, col_action, chance_actions);

                    for (auto chance_action : chance_actions)
                    {
                        typename Types::State state_copy = state;
                        state_copy.apply_actions(row_action, col_action, chance_action);
                        MatrixNode *matrix_node_next = chance_node->access(state_copy.get_obs());
                        assert(state_copy.get_obs() == chance_action);
                        matrix_node_next->stats.depth = stats.depth + 1;
                        matrix_node_next->stats.prob = state_copy.prob;
                        chance_node->stats.chance_strategy.push_back(state_copy.prob);

                        run_(max_depth, &state_copy, model_ptr, matrix_node_next);

                        stats.nash_payoff_matrix.get(row_idx, col_idx) +=
                            matrix_node_next->stats.payoff *
                            typename Types::Real{matrix_node_next->stats.prob};
                        stats.matrix_node_count += matrix_node_next->stats.matrix_node_count;
                    }

                    chance_node->stats.is_solved = true;
                    chance_node->stats.mutex.unlock();
                }
            }

            stats.payoff = LRSNash::solve(stats.nash_payoff_matrix, stats.row_solution, stats.col_solution);
            math::canonicalize(stats.payoff);
        }
    };
};
