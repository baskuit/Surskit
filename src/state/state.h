#pragma once

#include <types/types.h>

#include <concepts>
#include <vector>

template <typename Types>
concept IsStateTypes =
    requires(
        typename Types::State &state,
        const typename Types::State &const_state,
        typename Types::Action &action,
        typename Types::VectorAction &actions,
        typename Types::PRNG &device) {
        {
            const_state.is_terminal()
        } -> std::same_as<bool>;
        {
            const_state.get_payoff()
        } -> std::same_as<typename Types::Value>;
        {
            state.apply_actions(action, action)
        } -> std::same_as<void>;
        {
            state.get_actions(actions, actions)
        } -> std::same_as<void>;
        {
            state.randomize_transition(device)
        } -> std::same_as<void>;
        {
            const_state.get_obs()
        } -> std::same_as<const typename Types::Obs &>;
    } &&
    IsTypeList<Types>;

template <typename Types>
concept IsPerfectInfoStateTypes =
    requires(
        typename Types::State &state,
        const typename Types::State &const_state) {
        {
            state.row_actions
        } -> std::same_as<typename Types::VectorAction &>;
        {
            state.col_actions
        } -> std::same_as<typename Types::VectorAction &>;
        {
            state.prob
        } -> std::same_as<typename Types::Prob &>;
        {
            state.get_actions()
        } -> std::same_as<void>;
    } &&
    IsStateTypes<Types>;

template <IsTypeList Types>
class PerfectInfoState
{
public:
    bool terminal{false};
    Types::VectorAction row_actions;
    Types::VectorAction col_actions;
    Types::Value payoff;
    Types::Obs obs;
    Types::Prob prob;

    inline Types::Value get_payoff() const
    {
        return payoff;
    }

    inline bool is_terminal() const
    {
        return terminal;
    }

    inline const Types::Obs & get_obs() const {
        return obs;
    }

    inline void init_range_actions(size_t rows, size_t cols)
    {
        row_actions.resize(rows);
        col_actions.resize(cols);
        for (int i = 0; i < rows; ++i)
        {
            this->row_actions[i] = i;
        }
        for (int i = 0; i < cols; ++i)
        {
            this->col_actions[i] = i;
        }
    }

    inline void init_range_actions(size_t size)
    {
        row_actions.resize(size);
        col_actions.resize(size);
        for (int i = 0; i < size; ++i)
        {
            this->row_actions[i] = i;
            this->col_actions[i] = i;
        }
    }
};

template <typename Types>
concept IsChanceStateTypes =
    requires(
        typename Types::State &state,
        typename Types::Action &action,
        typename Types::Obs &obs,
        std::vector<typename Types::Obs> &chance_actions) {
        {
            state.get_chance_actions(action, action, chance_actions)
        } -> std::same_as<void>;
        {
            state.apply_actions(action, action, obs)
        } -> std::same_as<void>;
    } &&
    IsPerfectInfoStateTypes<Types>;

template <typename Types>
concept IsSolvedStateTypes =
    requires(
        typename Types::State state,
        typename Types::VectorReal &strategy,
        typename Types::MatrixValue &matrix) {
        {
            state.get_strategies(strategy, strategy)
        } -> std::same_as<void>;
        // {
        //     state.get_matrix(matrix)
        // } -> std::same_as<void>;
    } &&
    IsChanceStateTypes<Types>;
