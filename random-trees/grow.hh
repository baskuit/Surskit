#pragma once

#include <string>

#include "libsurskit/gambit.hh"
#include "model/model.hh"
#include "tree/tree.hh"
#include "algorithm/algorithm.hh"

/*
    This algorithm expands a node into a tree that is one-to-one with the abstract game tree
    and iteratively solves it.
*/

template <typename Model>
class Grow : public AbstractAlgorithm<Model>
{
    static_assert(std::derived_from<typename Model::Types::State, StateChance<typename Model::Types::TypeList>>,
        "This algorithm must be based on State type derived from StateChance");
    static_assert(std::derived_from<Model, DoubleOracleModel<typename Model::Types::State>>,
        "The Inference type of the DoubleOracleModel is used to store Nash strategies and payoff");

public:
    struct MatrixStats;
    struct ChanceStats;
    struct Types : AbstractAlgorithm<Model>::Types
    {
        using MatrixStats = Grow::MatrixStats;
        using ChanceStats = Grow::ChanceStats;
    };
    struct MatrixStats : AbstractAlgorithm<Model>::MatrixStats
    {
        typename Types::MatrixReal nash_payoff_matrix;
        int matrix_node_count = 1;
        int depth = 0;
    };
    struct ChanceStats : AbstractAlgorithm<Model>::ChanceStats
    {
        std::vector<typename Types::Observation> chance_actions; // we use vector here since this is general
        std::vector<typename Types::Probability> transition_probs;
    };

    int max_depth = -1;

    Grow() {}

    void grow(
        typename Types::State &state,
        Model &model,
        MatrixNode<Grow> *matrix_node)
    {

        // expand node
        state.get_actions();
        matrix_node->actions = state.actions;
        matrix_node->is_expanded = true;

        if (state.is_terminal)
        {
            matrix_node->inference.row_value = state.row_payoff;
            matrix_node->inference.col_value = state.col_payoff;
            matrix_node->is_terminal = true;
            return;
        }
        if (max_depth > 0 && matrix_node->stats.depth >= max_depth) 
        {
            model.get_inference(state, matrix_node->inference);
            matrix_node->is_terminal = true;
            return;
        }

        const int rows = state.actions.rows;
        const int cols = state.actions.cols;

        matrix_node->stats.nash_payoff_matrix.rows = rows;
        matrix_node->stats.nash_payoff_matrix.cols = cols;
        matrix_node->stats.nash_payoff_matrix.fill(rows, cols); // TODO change matrix fill methods to change rows, cols automatically.

        // recurse
        for (int row_idx = 0; row_idx < rows; ++row_idx)
        {
            for (int col_idx = 0; col_idx < cols; ++col_idx)
            {
                const typename Types::Action row_action = matrix_node->actions.row_actions[row_idx];
                const typename Types::Action col_action = matrix_node->actions.col_actions[col_idx];

                std::vector<typename Types::Observation> chance_actions;
                state.get_chance_actions(chance_actions, row_action, col_action);

                ChanceNode<Grow> *chance_node = matrix_node->access(row_idx, col_idx);
                for (auto chance_action : chance_actions)
                {
                    typename Types::State state_copy = state;
                    state_copy.apply_actions(row_action, col_action, chance_action);
                    MatrixNode<Grow> *matrix_node_next = chance_node->access(state_copy.transition); // TODO make sure seed state is getting transition object updated
                    matrix_node_next->stats.depth = matrix_node->stats.depth + 1;

                    grow(state_copy, model, matrix_node_next);

                    matrix_node->stats.nash_payoff_matrix.get(row_idx, col_idx) += matrix_node_next->inference.row_value * matrix_node_next->transition.prob;
                    matrix_node->stats.matrix_node_count += matrix_node_next->stats.matrix_node_count;
                }
            }
        }

        LibGambit::solve_matrix<Types>(
            matrix_node->stats.nash_payoff_matrix,
            matrix_node->inference.row_policy,
            matrix_node->inference.col_policy);
        for (int row_idx = 0; row_idx < rows; ++row_idx)
        {
            for (int col_idx = 0; col_idx < cols; ++col_idx)
            {
                matrix_node->inference.row_value +=
                    matrix_node->inference.row_policy[row_idx] *
                    matrix_node->inference.col_policy[col_idx] *
                    matrix_node->stats.nash_payoff_matrix.get(row_idx, col_idx);
                matrix_node->inference.col_value = 1 - matrix_node->inference.row_value;
            }
        }
        return;
    }

};