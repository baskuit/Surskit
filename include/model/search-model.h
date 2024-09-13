#pragma once

#include <model/model.h>
#include <algorithm/algorithm.h>
#include <tree/tree.h>

/*

Model + Search wrapped as a new model.

*/

namespace SearchModelDetail
{

    template <typename Types, bool use_policy>
    struct ModelOutputImpl;

    template <typename Types>
    struct ModelOutputImpl<Types, true>
    {
        Types::Value value;
        Types::VectorReal row_policy, col_policy;
    };

    template <typename Types>
    struct ModelOutputImpl<Types, false>
    {
        Types::Value value;
    };

};

template <
    IsSearchTypes Types,
    bool use_iterations = true,
    bool use_policy = true,
    bool use_tree_bandit = true,
    bool use_value = true,
    bool use_empirical = true>
struct SearchModel : Types::TypeList
{
    using State = typename Types::State; // DONT REMOVE
    // so State = is rebuilding the type list, in this case SearchModel isStateTypes.

    using ModelOutput = SearchModelDetail::ModelOutputImpl<Types, use_policy>;

    using ModelBatchInput = std::vector<typename Types::State>;
    using ModelBatchOutput = std::vector<typename Types::ModelOutput>;

    class Model
    {
    public:
        // time parameter (ms), number of iterations, or depth (in the case of solvers)
        const size_t count;
        Types::PRNG device;
        Types::Model model;
        Types::Search search;

        Model(
            const size_t count,
            const Types::PRNG &device,
            const Types::Model &model,
            const Types::Search &search)
            : count{count}, device{device}, model{model}, search{search}
        {
        }

        void inference(
            Types::State &&state,
            ModelOutput &output)
        {
            typename Types::MatrixNode root{};

            if constexpr (use_iterations)
            {
                search.run_for_iterations(count, device, state, model, root);
            }
            else
            {
                // as ms
                search.run(count, device, state, model, root);
            }

            if constexpr (use_policy)
            {
                if constexpr (use_tree_bandit)
                {
                    if constexpr (use_empirical) 
                    {
                        search.get_empirical_strategies(root.stats, output.row_policy, output.col_policy);
                    }
                    else
                    {
                        search.get_refined_strategies(root.stats, output.row_policy, output.col_policy);
                    }
                }
                else
                {
                    output.row_policy = root.row_solution;
                    output.col_policy = root.col_solution;
                }
                // std::cout << search << " : " << root.count_matrix_nodes() << std::endl;
            }
            if constexpr (use_value) {
                if constexpr (use_tree_bandit)
                {
                    search.get_empirical_value(root.stats, output.value);
                }
                else
                {
                    typename Types::Real x = (root.alpha + root.beta) / 2;
                    math::canonicalize(x); 
                    output.value = {x};
                }
            }
        }

        void inference(
            ModelBatchInput &batch_input,
            ModelBatchOutput &batch_output)
        {
            batch_output.resize(batch_input.size());
            for (size_t i = 0; i < batch_input.size(); ++i)
            {
                inference(batch_input[i], batch_output[i]);
            }
        }

        void add_to_batch_input(
            Types::State &&state,
            ModelBatchInput &batch_input) const
        {
            batch_input.push_back(state);
        }
    };
};
