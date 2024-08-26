#pragma once

#include <state/state.h>

/*
 Large uniform tree for testing etc. So called because it grows until it can't.
*/

template <typename Types = SimpleTypes>
struct MoldState : Types
{

    class State : public PerfectInfoState<Types>
    {
    public:
        size_t max_depth = 1;

        State(const size_t n_actions, const size_t max_depth) : max_depth{max_depth}
        {
            this->terminal = (this->max_depth == 0);
            this->init_range_actions(n_actions);
            this->prob = typename Types::Prob{1};
            this->obs = typename Types::Obs{};
        }

        void randomize_transition(
            Types::PRNG &device) const
        {
        }

        void get_actions() const
        {
        }

        void get_actions(
            Types::VectorAction &row_actions,
            Types::VectorAction &col_actions) const
        {
            row_actions = this->row_actions;
            col_actions = this->col_actions;
        }

        void apply_actions(
            Types::Action,
            Types::Action)
        {
            --this->max_depth;
            this->terminal = (this->max_depth == 0);
        }

        void apply_actions(
            Types::Action,
            Types::Action,
            Types::Obs)
        {
            --this->max_depth;
            this->terminal = (this->max_depth == 0);
        }

        void get_chance_actions(
            Types::Action,
            Types::Action,
            std::vector<typename Types::Obs> &chance_actions) const
        {
            chance_actions.resize(1);
        }
    };
};

struct OneSumMatrixGame : SimpleTypes
{
    class State : public PerfectInfoState<SimpleTypes>
    {
    public:
        const static int den = 10;
        SimpleTypes::MatrixValue payoff_matrix;

        State(const SimpleTypes::PRNG &device, size_t rows, size_t cols)
        {
            this->prob = SimpleTypes::Prob{1};
            auto device_copy = device;
            this->init_range_actions(rows, cols);
            payoff_matrix.fill(rows, cols);
            for (int i = 0; i < rows * cols; ++i)
            {
                int x = device_copy.random_int(den) + 1;
                SimpleTypes::Q row_payoff{x, den};
                SimpleTypes::Q col_payoff{den - x, den};
                payoff_matrix[i] = SimpleTypes::Value{row_payoff, col_payoff};
            }
        }

        State(const SimpleTypes::MatrixValue &payoff_matrix) : payoff_matrix{payoff_matrix} {}

        void randomize_transition(
            SimpleTypes::PRNG &device) const
        {
        }

        void get_actions() const
        {
        }

        void get_actions(
            SimpleTypes::VectorAction &row_actions,
            SimpleTypes::VectorAction &col_actions) const
        {
            row_actions = this->row_actions;
            col_actions = this->col_actions;
        }

        void apply_actions(
            SimpleTypes::Action row_action,
            SimpleTypes::Action col_action)
        {
            this->terminal = true;
            this->payoff = payoff_matrix.get(static_cast<int>(row_action), static_cast<int>(col_action));
        }

        void get_chance_actions(
            SimpleTypes::Action,
            SimpleTypes::Action,
            std::vector<SimpleTypes::Obs> &chance_actions) const
        {
            chance_actions.resize(1);
        }

        void apply_actions(
            SimpleTypes::Action row_action,
            SimpleTypes::Action col_action,
            SimpleTypes::Obs)
        {
            this->terminal = true;
            this->payoff = payoff_matrix.get(static_cast<int>(row_action), static_cast<int>(col_action));
        }
    };
};
