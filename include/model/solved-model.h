#pragma once

#include <state/state.h>

template <IsSolvedStateTypes Types>
struct SolvedStateModel : Types
{

    struct ModelOutput
    {
        Types::Value value;
        Types::VectorReal row_policy, col_policy;
    };
    using ModelBatchInput = std::vector<typename Types::State>;
    using ModelBatchOutput = std::vector<ModelOutput>;

    class Model
    {
    public:
        void inference(
            Types::State &&input,
            ModelOutput &output) const
        {
            output.value = input.get_payoff();
            input.get_strategies(output.row_policy, output.col_policy);
        }

        // TODO maybe faze out the use of Off Policy
        // Its useless and mostly confusing 
        
        // void inference(
        //     ModelBatchInput &batch_input,
        //     ModelBatchOutput &batch_output)
        // {
        //     for (auto &output : batch_output)
        //     {
        //         output.value = typename Types::Value{.5, .5};
        //     }
        // }

        // void add_to_batch_input(
        //     Types::State &&state,
        //     ModelBatchInput &batch_input) const
        // {
        //     batch_input.push_back(state);
        // }
    };
};
