#pragma once

#include <types/random.h>

/*

More functions for creating random trees. TODO

*/

#include <cmath>

namespace GrowLib
{

    template <typename RandomTree>
    int alternating_constant_actions (RandomTree *state, int actions) {
        int actions_to_move = std::max(state->rows, state->cols);
        if (actions == 1) {
            return actions_to_move;
        } else {
            return 1;
        }
    }

    template <typename RandomTree, int min, int max>
    int alternating_uniform_random_actions (RandomTree *state, int actions) {
        int actions_to_move = std::max(state->rows, state->cols);
        if (actions == 1) {
            return state->device.uniform_int(std::max(min, 2), max + 1);
        } else {
            return 1;
        }
    }

};