#pragma once

#include <string>

#include <libsurskit/gambit.hh>
#include <model/model.hh>
#include <tree/tree.hh>
#include <algorithm/algorithm.hh>

/*
    This algorithm expands a node into a tree that is one-to-one with the abstract game tree
    and iteratively solves it.

    The tree that is generated by this algorithm can then be wrapped in a TraversedState,
    which is a way of turning any StateChance into a SolvedState
*/

template <typename Model>
class FullTraversal : public AbstractAlgorithm<Model>
{
    // static_assert(std::derived_from<typename Model::Types::State, ChanceState<typename Model::Types::TypeList>>,
    //     "This algorithm must be based on State type derived from ChanceState");

    // static_assert(std::derived_from<Model, DoubleOracleModel<typename Model::Types::State>>,
    //     "The Inference type of the DoubleOracleModel is used to store Nash strategies and payoff");

public:
    struct MatrixStats;
    struct ChanceStats;
    struct Types : AbstractAlgorithm<Model>::Types
    {
        using MatrixStats = FullTraversal::MatrixStats;
        using ChanceStats = FullTraversal::ChanceStats;
    };
    struct MatrixStats : AbstractAlgorithm<Model>::MatrixStats
    {
        typename Types::Value payoff;
        typename Types::VectorReal row_solution, col_solution;
        typename Types::MatrixValue nash_payoff_matrix;
        size_t matrix_node_count = 1;
        unsigned int depth = 0;
        typename Types::Probability prob;
    };
    struct ChanceStats : AbstractAlgorithm<Model>::ChanceStats
    {
        std::vector<typename Types::Observation> chance_actions; // we use vector here since this is general
        std::vector<typename Types::Probability> chance_strategy;
    };

    int max_depth = -1;

    FullTraversal() {}

    void run(
        typename Types::State &state,
        Model &model,
        MatrixNode<FullTraversal> *matrix_node)
    {

        // expand node
        state.get_actions();
        matrix_node->row_actions = state.row_actions;
        matrix_node->col_actions = state.col_actions;
        matrix_node->is_expanded = true;
        
        auto &stats = matrix_node->stats;
        stats.prob = state.prob;

        if (state.is_terminal)
        {
            stats.payoff = state.payoff;
            matrix_node->is_terminal = true;
            return;
        }
        if (max_depth > 0 && stats.depth >= max_depth) 
        {
            typename Types::ModelOutput model_output;
            model.get_inference(state, model_output);
            stats.payoff = model_output.value;
            matrix_node->is_terminal = true;
            return;
        }

        const int rows = state.row_actions.size();
        const int cols = state.col_actions.size();

        stats.nash_payoff_matrix.fill(rows, cols, 0);

        // recurse
        for (ActionIndex row_idx = 0; row_idx < rows; ++row_idx)
        {
            for (ActionIndex col_idx = 0; col_idx < cols; ++col_idx)
            {
                const typename Types::Action row_action = matrix_node->row_actions[row_idx];
                const typename Types::Action col_action = matrix_node->col_actions[col_idx];

                std::vector<typename Types::Observation> chance_actions;
                state.get_chance_actions(chance_actions, row_action, col_action);

                ChanceNode<FullTraversal> *chance_node = matrix_node->access(row_idx, col_idx);
                for (auto chance_action : chance_actions)
                {
                    typename Types::State state_copy = state;
                    state_copy.apply_actions(row_action, col_action, chance_action);
                    MatrixNode<FullTraversal> *matrix_node_next = chance_node->access(state_copy.obs);
                    matrix_node_next->stats.depth = stats.depth + 1;

                    run(state_copy, model, matrix_node_next);

                    stats.nash_payoff_matrix.get(row_idx, col_idx) += matrix_node_next->stats.payoff * matrix_node_next->stats.prob;
                    stats.matrix_node_count += matrix_node_next->stats.matrix_node_count;
                }
            }
        }

        stats.row_solution.fill(rows);
        stats.col_solution.fill(cols);

        LibGambit::solve_matrix<Types>(
            stats.nash_payoff_matrix,
            stats.row_solution,
            stats.col_solution);
        stats.payoff = 0;
        for (int row_idx = 0; row_idx < rows; ++row_idx)
        {
            for (int col_idx = 0; col_idx < cols; ++col_idx)
            {
                stats.payoff +=
                    stats.nash_payoff_matrix.get(row_idx, col_idx) *
                    (stats.row_solution[row_idx] * stats.col_solution[col_idx]);
                // stats.col_payoff = 1 - stats.row_payoff;
            }
        }
        return;
    }

}; // TODO TODO review changes!!!!!