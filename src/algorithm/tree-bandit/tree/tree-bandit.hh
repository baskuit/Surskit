#pragma once

#include <types/types.hh>
#include <algorithm/algorithm.hh>
#include <tree/tree.hh>

#include <chrono>

template <
    CONCEPT(IsBanditAlgorithmTypes, Types),
    template <typename...> typename NodePair = DefaultNodes,
    bool return_if_expand = true,
    bool use_leaf_value = true>
    requires IsNodeTypes<NodePair<Types, typename Types::MatrixStats, typename Types::ChanceStats>>
// will auto complete all all this + ::MatrixNode but not the alias decl :(
struct TreeBandit : Types
{
    using MatrixNode = NodePair<Types, typename Types::MatrixStats, typename Types::ChanceStats>::MatrixNode;
    using ChanceNode = NodePair<Types, typename Types::MatrixStats, typename Types::ChanceStats>::ChanceNode;
    class Search : public Types::BanditAlgorithm
    {
    public:
        using Types::BanditAlgorithm::BanditAlgorithm;

        Search(const Types::BanditAlgorithm &base) : Types::BanditAlgorithm{base} {}

        friend std::ostream &operator<<(std::ostream &os, const Search &search)
        {
            os << "TreeBandit - ";
            os << static_cast<typename Types::BanditAlgorithm>(search);
            os << " - " << NodePair<Types, typename Types::MatrixStats, typename Types::ChanceStats>{};
            return os;
        }

        size_t run(
            size_t duration_ms,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode &matrix_node) const
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            typename Types::ModelOutput model_output;
            size_t iterations = 0;
            for (; duration.count() < duration_ms; ++iterations)
            {
                typename Types::State state_copy = state;
                state_copy.randomize_transition(device);
                this->run_iteration(device, state_copy, model, &matrix_node, model_output);
                end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            }
            return iterations;
        }

        size_t run_for_iterations(
            const size_t iterations,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode &matrix_node) const
        {
            auto start = std::chrono::high_resolution_clock::now();
            typename Types::ModelOutput model_output;
            for (size_t iteration = 0; iteration < iterations; ++iteration)
            {
                typename Types::State state_copy = state;
                state_copy.randomize_transition(device);
                this->run_iteration(device, state_copy, model, &matrix_node, model_output);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            return duration.count();
        }

    protected:
        MatrixNode *run_iteration(
            Types::PRNG &device,
            Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node,
            Types::ModelOutput &model_output) const
        {
            if (state.is_terminal())
            {
                matrix_node->set_terminal();
                model_output.value = state.payoff;
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
                    this->expand(matrix_node->stats, rows, cols, model_output);

                    if constexpr (return_if_expand)
                    {
                        return matrix_node;
                    }
                }
                else
                {
                    typename Types::Outcome outcome;
                    this->select(device, matrix_node->stats, outcome);

                    state.apply_actions(
                        state.row_actions[outcome.row_idx],
                        state.col_actions[outcome.col_idx]);
                    state.get_actions();

                    ChanceNode *chance_node = matrix_node->access(outcome.row_idx, outcome.col_idx);
                    MatrixNode *matrix_node_next = chance_node->access(state.get_obs());

                    MatrixNode *matrix_node_leaf = run_iteration(device, state, model, matrix_node_next, model_output);

                    if constexpr (use_leaf_value)
                    {
                        outcome.value = model_output.value;
                    }
                    else
                    {
                        this->get_empirical_value(matrix_node_next->stats, outcome.value);
                    }

                    this->update_matrix_stats(matrix_node->stats, outcome);
                    this->update_chance_stats(chance_node->stats, outcome);
                    return matrix_node_leaf;
                }
            }
        }
    };
};
