#pragma once

#include <state/state.h>

#include <tree/tree-debug.h>

#include <model/model.h>
#include <memory>
#include <unordered_map>
/*

Expands a shared tree around the input state by brute-forcing chance nodes.
This tree only stores the witnessed chance actions and their emprical probs.

Much like `FullTraversal`, this is exactly a regular Types::State but with the shared data as well.
When we  call `apply_actions` and transition to the of the shared tree, we mark as terminal.

*/

namespace MappedStateDetail
{

    template <typename Types, bool empirical>
    struct ChanceStatsImpl;

    template <typename Types>
    struct ChanceStatsImpl<Types, false>
    {

        struct BranchData
        {
            Types::Seed seed;
            Types::Prob prob{};
        };

        std::unordered_map<
            typename Types::Obs,
            BranchData,
            typename Types::ObsHash>
            map{};
        Types::Prob prob{};
    };

    template <typename Types>
    struct ChanceStatsImpl<Types, true>
    {
        struct BranchData
        {
            Types::Seed seed;
            int count{};
        };

        std::unordered_map<
            typename Types::Obs,
            BranchData,
            typename Types::ObsHash>
            map{};
        int count{};
    };

};

template <
    IsValueModelTypes Types,
    bool empirical = true> // TODO fix
struct MappedState : Types::TypeList
{

    struct MatrixStats
    {
    };

    using ChanceStats = MappedStateDetail::ChanceStatsImpl<Types, empirical>;

    using NodeTypes = DefaultNodes<
        Types,
        MatrixStats,
        ChanceStats,
        typename Types::VectorAction>; // actions are stored in the matrix node

    using MatrixNode = NodeTypes::MatrixNode;
    using ChanceNode = NodeTypes::ChanceNode;

    static void run(
        const size_t depth,
        const size_t tries,
        Types::PRNG &device,
        const Types::State &state,
        MatrixNode *matrix_node)
    {

        if (depth <= 0 || state.is_terminal())
        {
            matrix_node->set_terminal();
            return;
        }

        state.get_actions(matrix_node->row_actions, matrix_node->col_actions);

        for (int row_idx = 0; row_idx < matrix_node->row_actions.size(); ++row_idx)
        {
            for (int col_idx = 0; col_idx < matrix_node->col_actions.size(); ++col_idx)
            {
                ChanceNode *chance_node = matrix_node->access(row_idx, col_idx);

                for (int t = 0; t < tries; ++t)
                {
                    typename Types::State state_copy = state;
                    const typename Types::Seed seed{device.uniform_64()};
                    state_copy.randomize_transition(seed);
                    state_copy.apply_actions(
                        matrix_node->row_actions[row_idx],
                        matrix_node->col_actions[col_idx]);

                    MatrixNode *matrix_node_next = chance_node->access(state_copy.get_obs());
                    auto &branch_data = chance_node->stats.map[state_copy.get_obs()];

                    if constexpr (empirical)
                    {
                        if (branch_data.count == 0)
                        {
                            branch_data.seed = seed;
                            run(
                                depth - 1,
                                tries,
                                device,
                                state_copy,
                                matrix_node_next);
                        }
                        ++branch_data.count;
                        ++chance_node->stats.count;
                    }
                    else
                    {
                        if (branch_data.prob == 0) // TODO assumes no valid transition has 0 prob; rule on this!
                        {
                            branch_data.prob = state_copy.prob;
                            branch_data.seed = seed;
                            run(
                                depth - 1,
                                tries,
                                device,
                                state_copy,
                                matrix_node_next);
                        }
                        chance_node->stats.prob += state_copy.prob;
                    }
                }
            }
        }
    }

    class State : public Types::State
    {
    public:
        const MatrixNode
            *node;
        std::shared_ptr<const MatrixNode>
            explored_tree;
        const Types::Model
            model;

        template <typename... T>
        State(
            const size_t depth,
            const size_t tries,
            const Types::PRNG &device,
            const Types::Model &model,
            const T &...args)
            : Types::State{args...},
              model{model}
        {
            typename Types::PRNG device_{device};
            auto temp_tree = std::make_shared<MatrixNode>();
            node = temp_tree.get();
            run(depth, tries, device_, *this, temp_tree.get());
            explored_tree = temp_tree;
            this->row_actions = node->row_actions;
            this->col_actions = node->col_actions;
        }

        void get_actions()
        {
            this->row_actions = node->row_actions;
            this->col_actions = node->col_actions;
        }

        void get_actions(
            Types::VectorAction row_actions,
            Types::VectorAction col_actions) const
        {
            row_actions = node->row_actions;
            col_actions = node->col_actions;
        }

        void get_chance_actions(
            Types::Action row_action,
            Types::Action col_action,
            std::vector<typename Types::Obs> &chance_actions) const
        {
            const int row_idx = std::find(node->row_actions.begin(), node->row_actions.end(), row_action) - node->row_actions.begin();
            const int col_idx = std::find(node->col_actions.begin(), node->col_actions.end(), col_action) - node->col_actions.begin();
            const ChanceNode *chance_node = node->access(row_idx, col_idx);
            // TODO sort by prob for faster AlphaBeta
            for (auto kv : chance_node->stats.map)
            {
                chance_actions.push_back(kv.first);
            }
        }

        void apply_actions(
            Types::Action row_action,
            Types::Action col_action)
        {
            // only appropriate for solves, which don't use this method
            // risk is that you encounter and unseen transition
            // make current node nullptr?. unconvincing
            std::exception();
        }

        void apply_actions(
            Types::Action row_action,
            Types::Action col_action,
            Types::Obs chance_action)
        {
            const int row_idx = std::find(node->row_actions.begin(), node->row_actions.end(), row_action) - node->row_actions.begin();
            const int col_idx = std::find(node->col_actions.begin(), node->col_actions.end(), col_action) - node->col_actions.begin();
            const ChanceNode *chance_node = node->access(row_idx, col_idx);
            const auto &branch_data = chance_node->stats.map.at(chance_action);
            Types::State::randomize_transition(branch_data.seed);
            Types::State::apply_actions(row_action, col_action);
            node = chance_node->access(Types::State::get_obs());
            // check seed took us to the same place
            assert(chance_action == Types::State::get_obs());
            if constexpr (empirical)
            {
                this->prob = typename Types::Prob{
                    typename Types::Q{static_cast<int>(branch_data.count), static_cast<int>(chance_node->stats.count)}};
            }
            else
            {
                this->prob = branch_data.prob / chance_node->stats.prob;
            }
            this->row_actions = node->row_actions;
            this->col_actions = node->col_actions;
        }

        bool is_terminal() const
        {
            return node->is_terminal();
        }

        Types::Value get_payoff() const
        {
            if (Types::State::is_terminal())
            {
                return Types::State::get_payoff();
            }
            typename Types::State state_{*this};
            state_.get_actions();
            typename Types::Model model_{model};
            typename Types::ModelOutput output;
            model_.inference(std::move(state_), output);
            return output.value;
        }
    };
};