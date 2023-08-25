#pragma once

#include <string>

#include <libsurskit/lrslib.hh>
#include <model/model.hh>
#include <tree/tree.hh>
#include <algorithm/algorithm.hh>

/*
    This algorithm expands a node into a tree that is one-to-one with the abstract game tree
    and iteratively solves it.

    The tree that is generated by this algorithm can then be wrapped in a TraversedState,
    which is a way of turning any StateChance into a SolvedState
*/

template <
    CONCEPT(IsValueModelTypes, Types),
    template <typename...> typename NodePair = DefaultNodes>
requires IsChanceStateTypes<Types> 
struct FullTraversal : Types
{
    struct MatrixStats
    {
        typename Types::Value payoff{};
        typename Types::VectorReal row_solution, col_solution;
        typename Types::MatrixValue nash_payoff_matrix;

        size_t matrix_node_count = 1;
        unsigned int depth = 0;
        typename Types::Prob prob;
    };
    struct ChanceStats
    {
        std::vector<typename Types::Obs> chance_actions;
        std::vector<typename Types::Prob> chance_strategy;
    };
    using MatrixNode = typename NodePair<Types, MatrixStats, ChanceStats>::MatrixNode;
    using ChanceNode = typename NodePair<Types, MatrixStats, ChanceStats>::ChanceNode;

    class Search
    {
    public:
        const int max_depth = -1;

        Search() {}
        Search(const int max_depth) : max_depth{max_depth} {}

        void run(
            const Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node) const
        {
            auto state_ = state;
            run_(state_, model, matrix_node);
        }

        void run_(
            Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node) const
        {
            // expand node
            state.get_actions();
            matrix_node->expand(state);

            MatrixStats &stats = matrix_node->stats;

            if (state.is_terminal())
            {
                stats.payoff = state.payoff;
                matrix_node->set_terminal();
                return;
            }
            if (max_depth > 0 && stats.depth >= max_depth)
            {
                model.get_value(state, stats.payoff);
                matrix_node->set_terminal();
                return;
            }

            const int rows = state.row_actions.size();
            const int cols = state.col_actions.size();

            stats.nash_payoff_matrix.fill(rows, cols);
            stats.row_solution.resize(rows);
            stats.col_solution.resize(cols);

            // recurse
            for (int row_idx = 0; row_idx < rows; ++row_idx)
            {
                for (int col_idx = 0; col_idx < cols; ++col_idx)
                {
                    const typename Types::Action &row_action{state.row_actions[row_idx]};
                    const typename Types::Action &col_action{state.col_actions[col_idx]};
                    
                    ChanceNode *chance_node = matrix_node->access(row_idx, col_idx);
                    auto &chance_actions = chance_node->stats.chance_actions;
                    state.get_chance_actions(row_action, col_action, chance_actions);

                    for (auto chance_action : chance_actions)
                    {
                        typename Types::State state_copy = state;
                        state_copy.apply_actions(row_action, col_action, chance_action);
                        // state.apply_fast(row_idx, row_actions, col_idx, col_actions);
                        // {
                        //     state.apply_actions(row_idx, col_idx);
                        // }
                        MatrixNode *matrix_node_next = chance_node->access(state_copy.obs);
                        matrix_node_next->stats.depth = stats.depth + 1;
                        matrix_node_next->stats.prob = state_copy.prob;
                        chance_node->stats.chance_strategy.push_back(state_copy.prob);

                        run_(state_copy, model, matrix_node_next);

                        stats.nash_payoff_matrix.get(row_idx, col_idx) += 
                            matrix_node_next->stats.payoff * 
                            typename Types::Real{matrix_node_next->stats.prob};
                        stats.matrix_node_count += matrix_node_next->stats.matrix_node_count;
                    }
                }
            }

            auto value_pair = LRSNash::solve(stats.nash_payoff_matrix, stats.row_solution, stats.col_solution);
            stats.payoff = typename Types::Value{value_pair};
        }
    };
};