#include <unordered_map>

template <
    typename Types,
    typename Options = SearchOptions<>>
struct TreeBanditFlat : Types
{
    template <typename MatrixStats, typename NodeActions, typename NodeValue>
    struct MData
    {
        NodeActions row_actions, col_actions;
        MatrixStats stats;
        NodeValue value;
    };

    template <typename MatrixStats, typename NodeActions>
    struct MData<MatrixStats, NodeActions, void>
    {
        NodeActions row_actions, col_actions;
        MatrixStats stats;
    };

    template <typename MatrixStats, typename NodeValue>
    struct MData<MatrixStats, void, NodeValue>
    {
        MatrixStats stats;
        NodeValue value;
    };

    template <typename MatrixStats>
    struct MData<MatrixStats, void, void>
    {
        MatrixStats stats;
    };

    using MatrixData = MData<typename Types::MatrixStats, typename Options::NodeActions, typename Options::NodeValue>;

    class Search : public Types::BanditAlgorithm
    {
    public:
        using Types::BanditAlgorithm::BanditAlgorithm;

        Search(const Types::BanditAlgorithm &base) : Types::BanditAlgorithm{base} {}

        std::array<MatrixData, Options::max_iterations> matrix_data{};

        // reset every run call
        size_t iteration = 0;

        bool info[2 * Options::max_iterations];

        std::unordered_map<uint64_t, int> transition{};

        // reset every iteration
        int depth = 0;
        int index = 0;
        typename Types::ModelOutput leaf_output;
        int rows, cols;

        const Types::ObsHash hash_function{};

        std::array<typename Types::Outcome, Options::max_depth> outcomes{};

        std::array<int, Options::max_depth> matrix_indices{};

        size_t run_for_iterations(
            const size_t iterations,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model)
        {
            transition = std::unordered_map<uint64_t, int>{};
            memset(info, false, 2 * Options::max_iterations * sizeof(bool));
            const auto start = std::chrono::high_resolution_clock::now();
            for (iteration = 0; iteration < iterations; ++iteration)
            {
                typename Types::State state_copy = state;
                state_copy.randomize_transition(device);
                run_iteration(device, state_copy, model);
            }
            const auto end = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            return duration.count();
        }

        void run_iteration(
            Types::PRNG &device,
            Types::State &state,
            Types::Model &model)
        {
            depth = 0;
            index = 0;

            // side by side: was_seen and is_expanded (state, e.g. expanded vectors)
            bool *info_ptr = &info[2 * index];

            while (*info_ptr && !state.is_terminal() && depth < Options::max_depth)
            {
                typename Types::Outcome &outcome = outcomes[depth];
                typename Types::MatrixStats &stats = matrix_data[index].stats;
                MatrixData &current_data = matrix_data[index];

                // not really expanded
                if (!*(info_ptr + 1))
                {
                    if constexpr (!std::is_same_v<typename Options::NodeActions, void>)
                    {
                        state.get_actions(matrix_data[index].row_actions, current_data.col_actions);
                        rows = current_data.row_actions.size();
                        cols = current_data.col_actions.size();
                    }
                    else
                    {
                        rows = state.row_actions.size();
                        cols = state.col_actions.size();
                    }
                    this->expand_state_part(stats, rows, cols);
                    *(info_ptr + 1) = true;
                }

                this->select(device, stats, outcome);

                if constexpr (!std::is_same_v<typename Options::NodeActions, void>)
                {
                    state.apply_actions(
                        current_data.row_actions[outcome.row_idx],
                        current_data.col_actions[outcome.col_idx]);
                }
                else
                {
                    state.apply_actions(
                        state.row_actions[outcome.row_idx],
                        state.col_actions[outcome.col_idx]);
                    state.get_actions();
                }

                ++depth;
                uint64_t hash_ = hash(index, outcome.row_idx, outcome.col_idx, hash_function(state.get_obs()));

                if (transition[hash_] == 0)
                {
                    index = iteration;
                    transition[hash_] = iteration;
                    info_ptr = &info[2 * index];
                    if constexpr (std::is_same_v<typename Options::return_after_expand, void>)
                    {
                        break;
                    }
                }
                else
                {
                    index = transition[hash_];
                    info_ptr = &info[2 * index];
                }
                matrix_indices[depth] = index;
            }

            if (state.is_terminal())
            {
                leaf_output.value = state.get_payoff();
            }
            else
            {
                *info_ptr = true;
                model.inference(std::move(state), leaf_output);
            }

            this->expand_inference_part(matrix_data[index].stats, leaf_output);
            if constexpr (!std::is_same_v<typename Options::NodeValue, void>)
            {
                matrix_data[index].value = leaf_output.value;
            }

            for (int d = 0; d < depth; ++d)
            {
                if constexpr (std::is_same_v<typename Options::update_using_average, void>)
                {
                    outcomes[d].value = leaf_output.value;
                }
                else
                {
                    // TODO fix!! check d + 1
                    this->get_empirical_value(matrix_data[matrix_indices[d + 1]].stats, outcomes[d].value);
                }

                this->update_matrix_stats(matrix_data[matrix_indices[d]].stats, outcomes[d]);
            }
        }

        inline uint64_t hash(
            const int index,
            const int row_idx,
            const int col_idx,
            const uint64_t obs_hash) const
        {
            size_t h = 17;
            h = h * 31 + index;
            h = h * 31 + row_idx;
            h = h * 31 + col_idx;
            h = h * 31 + static_cast<uint64_t>(obs_hash);
            h = h * 31 + static_cast<uint64_t>(obs_hash >> 32);
            return h;
        }
    };
};
