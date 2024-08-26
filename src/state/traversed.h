#pragma once

#include <model/model.h>
#include <algorithm/solver/full-traversal.h>
#include <tree/tree.h>
#include <tree/tree-debug.h>

#include <memory>

/*

Takes a model type list (for FullTraversal) and defines a State that is
just a Types::State (literally, it is derived) but with the extra information of a
shared search tree and node address in the tree.

All methods are inherited/stay the same, except for get_actions(...)
since those need to update the matrix node pointer.
We also add the get_stategies and get_matrix methods required by the IsSolvedStateTypes concept

*/

template <
    IsValueModelTypes Types>
    requires IsChanceStateTypes<Types>
struct TraversedState : Types::TypeList
{
    template <IsNodeTypes NodePair>
    class StateWithNodes;

    using State =
        StateWithNodes<
            DebugNodes<
                Types,
                typename FullTraversal<Types, DebugNodes>::MatrixStats,
                typename FullTraversal<Types, DebugNodes>::ChanceStats>>;

    // This hidden template impl allows for type hints
    template <IsNodeTypes NodePair>
    class StateWithNodes : public Types::State
    {
    public:
        const typename NodePair::MatrixNode
            *node;
        std::shared_ptr<const typename NodePair::MatrixNode>
            full_traversal_tree;

        StateWithNodes(
            const Types::State &state,
            Types::Model &model,
            int max_depth = -1)
            : Types::State{state}
        {
            auto temp_tree = std::make_shared<typename NodePair::MatrixNode>();
            node = temp_tree.get();
            auto search = typename FullTraversal<Types, DebugNodes>::Search{};
            search.run(max_depth, *this, model, *temp_tree.get());
            full_traversal_tree = temp_tree;
        }

        void apply_actions(
            Types::Action row_action,
            Types::Action col_action)
        {
            Types::State::apply_actions(row_action, col_action);
            const int row_idx = std::find(this->row_actions.begin(), this->row_actions.end(), row_action) - this->row_actions.begin();
            const int col_idx = std::find(this->col_actions.begin(), this->col_actions.end(), col_action) - this->col_actions.begin();
            node = node->access(row_idx, col_idx)->access(this->obs);
            // this method uses the const access function
        }

        void apply_actions(
            Types::Action row_action,
            Types::Action col_action,
            Types::Obs chance_action)
        {
            Types::State::apply_actions(row_action, col_action, chance_action);
            const int row_idx = std::find(this->row_actions.begin(), this->row_actions.end(), row_action) - this->row_actions.begin();
            const int col_idx = std::find(this->col_actions.begin(), this->col_actions.end(), col_action) - this->col_actions.begin();
            node = node->access(row_idx, col_idx)->access(this->obs);
        }

        void get_strategies(
            Types::VectorReal &row_strategy,
            Types::VectorReal &col_strategy) const
        {
            row_strategy = node->stats.row_solution;
            col_strategy = node->stats.col_solution;
        }

        void get_matrix(
            Types::MatrixValue &payoff_matrix) const
        {
            payoff_matrix = node->stats.nash_payoff_matrix;
        }
    };
};

/*
A version of the TraversedSimState type list where the new State type
is not inherited from and does not maintain a 'guide' value of the old State type
Instead it simulates the state via the tree stats.

It may be faster than using the TraversedState version
*/

template <IsValueModelTypes Types>
    requires IsChanceStateTypes<Types>
struct TraversedSimState : Types::TypeList
{
    template <IsNodeTypes NodePair>
    class State_;

    using State = State_<DefaultNodes<Types, typename FullTraversal<Types>::MatrixStats, typename FullTraversal<Types>::ChanceStats>>;

    template <IsNodeTypes NodePair>
    class State_ : public PerfectInfoState<typename Types::TypeList>
    {
    public:
        typename NodePair::MatrixNode const
            *node;
        std::shared_ptr<const typename NodePair::MatrixNode>
            full_traversal_tree;
        double chance_p;

        State_(
            Types::State &state,
            Types::Model &model,
            int max_depth = -1)
        {
            auto temp_tree = std::make_shared<typename NodePair::MatrixNode>();
            node = temp_tree.get();
            auto search = typename FullTraversal<Types>::Search{max_depth};
            search.run(*this, model, temp_tree.get());
            full_traversal_tree = temp_tree;
        }

        void get_actions()
        {
            this->row_actions = node->get_row_actions();
            this->col_actions = node->get_col_actions();
        }

        void get_actions(
            Types::VectorAction &row_actions,
            Types::VectorAction &col_actions) const
        {
            this->row_actions = node->get_row_actions();
            this->col_actions = node->get_col_actions();
        }

        void randomize_transition(
            Types::PRNG &device)
        {
            chance_p = device.uniform();
        }

        void apply_actions(
            Types::Action row_action,
            Types::Action col_action)
        {
            int row_idx = std::find(this->row_actions.begin(), this->row_actions.end(), row_action) - this->row_actions.begin();
            int col_idx = std::find(this->col_actions.begin(), this->col_actions.end(), col_action) - this->col_actions.begin();
            typename NodePair::ChanceNode *chance_node = node->access(row_idx, col_idx);
            auto &chance_strategy = chance_node->stats.chance_strategy;

            int chance_idx = 0;
            while (chance_p > 0 && chance_idx + 1 < chance_strategy.size())
            {
                chance_p -= static_cast<double>(chance_strategy[chance_idx]);
                ++chance_idx;
            }

            const typename Types::Obs chance_action = chance_node->stats.chance_actions[chance_idx];
            node = chance_node->access(chance_action);
            update_perfect_info_state();
        }

        void get_chance_actions(
            Types::Action row_action,
            Types::Action col_action,
            std::vector<typename Types::Obs> &chance_actions) const
        {
            int row_idx = std::find(this->row_actions.begin(), this->row_actions.end(), row_action) - this->row_actions.begin();
            int col_idx = std::find(this->col_actions.begin(), this->col_actions.end(), col_action) - this->col_actions.begin();
            typename NodePair::ChanceNode *chance_node = node->access(row_idx, col_idx);
            chance_actions = chance_node->stats.chance_actions;
        }

        void apply_actions(
            Types::Action row_action,
            Types::Action col_action,
            Types::Obs chance_action)
        {
            int row_idx = std::find(this->row_actions.begin(), this->row_actions.end(), row_action) - this->row_actions.begin();
            int col_idx = std::find(this->col_actions.begin(), this->col_actions.end(), col_action) - this->col_actions.begin();
            node = node->access(row_idx, col_idx)->access(chance_action);
            update_perfect_info_state();
        }

        void get_strategies(
            Types::VectorReal &row_strategy,
            Types::VectorReal &col_strategy) const
        {
            row_strategy = node->stats.row_solution;
            col_strategy = node->stats.col_solution;
        }

    private:
        void update_perfect_info_state()
        {
            this->terminal = node->is_terminal();
            this->payoff = node->stats.payoff;
        }
    };
};
