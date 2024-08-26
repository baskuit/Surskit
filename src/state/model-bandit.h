#include <libpinyon/dynamic-wrappers.h>
#include <types/types.h>
#include <algorithm/tree-bandit/tree/multithreaded.h>

#include <utility>

struct ModelBandit : SimpleTypes
{
    class State : public PerfectInfoState<SimpleTypes>
    {
    public:
        W::Types::State (*state_generator)(SimpleTypes::Seed){nullptr};
        std::vector<W::Types::Model> models{};
        const size_t vs_rounds;
        SimpleTypes::Seed state_seed{};

        State(
            W::Types::State (*state_generator)(SimpleTypes::Seed),
            const std::vector<W::Types::Model> &models,
            size_t vs_rounds = 1)
            : state_generator{state_generator}, models{models}, vs_rounds{vs_rounds}
        {
            this->init_range_actions(models.size());
        }

        void get_actions() const {}

        void get_actions(
            SimpleTypes::VectorAction &row_actions,
            SimpleTypes::VectorAction &col_actions) const
        {
            row_actions = this->row_actions;
            col_actions = this->col_actions;
        }

        void randomize_transition(SimpleTypes::PRNG &device)
        {
            state_seed = device.uniform_64();
        }

        void apply_actions(
            SimpleTypes::Action row_action,
            SimpleTypes::Action col_action)
        {
            if (row_action == col_action)
            {
                // skip if its a mirror match
                this->payoff = SimpleTypes::Value{.5, .5};
                this->obs = 0;
                this->terminal = true;
                return;
            }
            W::Types::PRNG device{state_seed};
            W::Types::Model row_model = models[static_cast<int>(row_action)];
            W::Types::Model col_model = models[static_cast<int>(col_action)];
            // copy constr will clone the unique_ptr member

            W::Types::State state = (*state_generator)(state_seed);
            W::Types::Value total_payoff{};
            for (int i = 0; i < vs_rounds; ++i)
            {
                W::Types::Value row_first_payoff = play_vs(device, row_model, col_model, state);
                W::Types::Value col_first_payoff = play_vs(device, col_model, row_model, state);
                W::Types::Value col_first_payoff_flipped{col_first_payoff.get_col_value(), col_first_payoff.get_row_value()};
                total_payoff += row_first_payoff;
                total_payoff += col_first_payoff_flipped;
            }
            this->payoff = SimpleTypes::Value{
                SimpleTypes::Real{total_payoff.get_row_value() / double(2 * vs_rounds)},
                SimpleTypes::Real{total_payoff.get_col_value() / double(2 * vs_rounds)}};
            this->terminal = true;
            this->obs = SimpleTypes::Obs{static_cast<int>(device.get_seed())};
        }

    private:
        W::Types::Value play_vs(
            W::Types::PRNG &device,
            W::Types::Model &row_model,
            W::Types::Model &col_model,
            const W::Types::State &state_) const
        {
            W::Types::State state = state_;
            W::Types::ModelOutput row_output, col_output;

            state.get_actions();
            int row_idx, col_idx;
            int turn = 0;
            while (!state.is_terminal())
            {
                row_model.inference(state, row_output);
                col_model.inference(state, col_output);
                row_idx = device.sample_pdf(row_output.row_policy);
                col_idx = device.sample_pdf(col_output.col_policy);

                state.apply_actions(row_idx, col_idx);
                state.get_actions();
                ++turn;
            }
            return state.get_payoff();
        }
    };
};
