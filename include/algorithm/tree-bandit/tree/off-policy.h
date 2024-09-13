#pragma once

#include <types/types.h>
#include <tree/tree.h>
#include <algorithm/algorithm.h>

template <
    IsBanditAlgorithmTypes Types,
    template <typename...> typename NodePair = DefaultNodes,
    bool return_if_expand = true> // when would this be not true?

struct OffPolicy : Types
{
    struct MatrixStats : Types::MatrixStats
    {
        bool properly_expanded = false;
    };
    struct ChanceStats : Types::ChanceStats
    {
    };
    using MatrixNode = NodePair<Types, MatrixStats, ChanceStats>::MatrixNode;
    using ChanceNode = NodePair<Types, MatrixStats, ChanceStats>::ChanceNode;

    struct Frame
    {
        Frame(MatrixNode *matrix_node) : matrix_node{matrix_node} {}

        Types::Outcome outcome;
        MatrixNode *matrix_node;
    };

    struct Trajectory
    {
        std::vector<Frame> frames;
    };

    class Search : public Types::BanditAlgorithm
    {
    public:
        using Types::BanditAlgorithm::BanditAlgorithm;

        Search(const Types::BanditAlgorithm &base) : Types::BanditAlgorithm{base} {}

        friend std::ostream &operator<<(std::ostream &os, const Search &search)
        {
            os << "OffPolicy - ";
            os << static_cast<typename Types::BanditAlgorithm>(search);
            os << " - " << NodePair<Types, typename Types::MatrixStats, typename Types::ChanceStats>{};
            return os;
        }

        size_t run(
            const size_t duration_ms,
            const size_t actor_iterations_per,
            Types::PRNG &device,
            const std::vector<typename Types::State> &states,
            Types::Model &model,
            std::vector<MatrixNode> &matrix_nodes)
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::vector<Trajectory> trajectories{};

            size_t iterations = 0;
            for (; duration.count() < duration_ms; ++iterations)
            {
                trajectories.clear();
                typename Types::ModelBatchInput model_batch_input{};
                get_trajectories(trajectories, model_batch_input,
                                 actor_iterations_per, device, states, model, matrix_nodes);

                typename Types::ModelBatchOutput model_batch_output{};
                model.inference(model_batch_input, model_batch_output);

                update_using_trajectories(model, trajectories, model_batch_output);
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            }
            return iterations;
        }

        size_t run_for_iterations(
            const size_t learner_iterations,
            const size_t actor_iterations_per,
            Types::PRNG &device,
            const std::vector<typename Types::State> &states,
            Types::Model &model,
            std::vector<MatrixNode> &matrix_nodes)
        {
            std::vector<Trajectory> trajectories{};

            auto start = std::chrono::high_resolution_clock::now();
            for (int learner_iteration = 0; learner_iteration < learner_iterations; ++learner_iteration)
            {
                trajectories.clear();
                typename Types::ModelBatchInput model_batch_input{};
                get_trajectories(trajectories, model_batch_input,
                                 actor_iterations_per, device, states, model, matrix_nodes);

                typename Types::ModelBatchOutput model_batch_output{};
                model.inference(model_batch_input, model_batch_output);

                update_using_trajectories(model, trajectories, model_batch_output);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            return duration.count();
        }

        void get_trajectories(
            // output parameters
            std::vector<Trajectory> &trajectories,
            Types::ModelBatchInput &input,
            // normal tree bandit params
            size_t actor_iterations_per,
            Types::PRNG &device,
            const std::vector<typename Types::State> &states,
            Types::Model &model,
            std::vector<MatrixNode> &matrix_nodes)
        {
            // One inference step
            for (int index = 0; index < matrix_nodes.size(); ++index)
            {
                MatrixNode *matrix_node = &matrix_nodes[index];
                const typename Types::State &state = states[index];
                // state, matrix node pair

                for (size_t actor_iteration = 0; actor_iteration < actor_iterations_per; ++actor_iteration)
                {
                    trajectories.emplace_back();
                    Trajectory &trajectory = trajectories.back();
                    // trajectory is a flat vector over matrix nodes and input iterations per

                    auto state_copy = state;
                    get_trajectory(trajectory,
                                   device, state_copy, model, matrix_node);
                    // populate vector of frames, rollout state to leaf node

                    if (!state_copy.is_terminal())
                    {
                        model.add_to_batch_input(std::move(state_copy), input);
                    }
                    // then add leaf state to inference pile
                }
            }
        }

        void update_using_trajectories(
            typename Types::Model &model,
            std::vector<Trajectory> &trajectories,
            Types::ModelBatchOutput &model_batch_output)
        {
            int index = 0;
            for (Trajectory &trajectory : trajectories)
            {
                Frame &leaf_frame = trajectory.frames.front();
                MatrixStats &leaf_stats = leaf_frame.matrix_node->stats;

                typename Types::Value leaf_value;
                if (leaf_frame.matrix_node->is_terminal()) [[unlikely]]
                {
                    leaf_value = leaf_frame.outcome.value;
                }
                else [[likely]]
                {
                    typename Types::ModelOutput model_output{};
                    model.get_output(model_output, model_batch_output, index);
                    leaf_value = model_output.value;
                    if (!leaf_stats.properly_expanded)
                    {
                        this->expand_inference_part(leaf_stats, model_output);
                        leaf_stats.properly_expanded = true;
                    }
                }

                int frame_index = 0;
                for (Frame &frame : trajectory.frames)
                {
                    if (frame_index == 0)
                    {
                        frame_index++;
                        continue;
                    } // skip last frame in trajectory

                    frame.outcome.value = leaf_value;
                    this->update_matrix_stats_offpolicy(
                        frame.matrix_node->stats,
                        frame.outcome);
                    // the difference is that the outcome row_mu entries reflect that of the 'actor'
                }
            }
        }

        void get_trajectory(
            // output
            Trajectory &trajectory,
            // normal run_iteration args
            Types::PRNG &device,
            Types::State &state,
            Types::Model &model,
            MatrixNode *matrix_node)
        {
            Frame frame{matrix_node};

            if (state.is_terminal())
            {
                frame.outcome.value = state.get_payoff();
                matrix_node->set_terminal();
            }
            else
            {
                if (!matrix_node->is_expanded())
                {
                    const size_t rows = state.row_actions.size();
                    const size_t cols = state.col_actions.size();
                    matrix_node->expand(rows, cols);
                    this->expand_state_part(matrix_node->stats, rows, cols);
                }
                else
                {
                    state.get_actions(); // get_actions called here before add_to_batch_input();
                    // we need this to have valid action data when it gets there, for policy masking
                    if (matrix_node->stats.properly_expanded)
                    {
                        this->select(device, matrix_node->stats, frame.outcome);
                    }
                    else
                    {
                        const int rows = state.row_actions.size();
                        const int cols = state.col_actions.size();
                        frame.outcome.row_idx = device.random_int(rows);
                        frame.outcome.col_idx = device.random_int(cols);
                        frame.outcome.row_mu = typename Types::Q{1, rows};
                        frame.outcome.col_mu = typename Types::Q{1, cols};
                    }
                    state.apply_actions(
                        state.row_actions[frame.outcome.row_idx],
                        state.col_actions[frame.outcome.col_idx]);
                    ChanceNode *chance_node = matrix_node->access(frame.outcome.row_idx, frame.outcome.col_idx);
                    MatrixNode *matrix_node_next = chance_node->access(state.get_obs());

                    get_trajectory(trajectory, device, state, model, matrix_node_next);
                }
            }

            trajectory.frames.push_back(frame);
        }
    };
};
