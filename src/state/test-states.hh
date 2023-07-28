#pragma once

#include <state/state.hh>

/*
 Large uniform tree for testing etc. So called because it grows until it can't.
*/

template <int size>
class MoldState : public PerfectInfoState<SimpleTypes>
{
public:
    struct Types : PerfectInfoState<SimpleTypes>::Types
    {
    };

    size_t max_depth = 1;

    MoldState(size_t max_depth) : max_depth((max_depth >= 0) * max_depth)
    {
        this->row_actions.resize(size);
        this->col_actions.resize(size);

        for (int i = 0; i < size; ++i)
        {
            this->row_actions[i] = typename Types::Action{i};
            this->col_actions[i] = typename Types::Action{i};
        }
        this->prob = typename Types::Probability{Rational<>{1}};
    }

    void reseed(typename Types::PRNG &device)
    {
    }

    void get_actions()
    {
        this->is_terminal = (this->max_depth <= 0);
    }

    void apply_actions(
        typename Types::Action row_action,
        typename Types::Action col_action)
    {
        --this->max_depth;
    }
};