#pragma once

#include <concepts>
#include <type_traits>

template <typename Types>
concept IsNodeTypes =
    requires(
        typename Types::MatrixNode &matrix_node,
        const typename Types::MatrixNode &const_matrix_node,
        typename Types::ChanceNode &chance_node,
        const typename Types::ChanceNode &const_chance_node,
        typename Types::Obs &obs,
        typename Types::Prob &prob,
        typename Types::State &state) {
        {
            matrix_node.stats
        } -> std::same_as<typename Types::MatrixStats &>;
        {
            matrix_node.access(0, 0)
        } -> std::same_as<typename Types::ChanceNode *>;
        {
            const_matrix_node.access(0, 0)
        } -> std::same_as<const typename Types::ChanceNode *>;
        {
            const_matrix_node.is_expanded()
        } -> std::same_as<bool>;
        {
            matrix_node.expand(0, 0)
        } -> std::same_as<void>;
        {
            const_matrix_node.is_terminal()
        } -> std::same_as<bool>;

        {
            chance_node.stats
        } -> std::same_as<typename Types::ChanceStats &>;
        {
            chance_node.access(obs)
        } -> std::same_as<typename Types::MatrixNode *>;
        {
            const_chance_node.access(obs)
        } -> std::same_as<const typename Types::MatrixNode *>;
    };

template <typename Types, typename Actions, typename Value>
struct MatrixNodeData
{
    Actions row_actions, col_actions;
    Value value;
};

template <typename Types, typename Actions>
struct MatrixNodeData<Types, Actions, void>
{
    Actions row_actions, col_actions;
};

template <typename Types, typename Value>
struct MatrixNodeData<Types, void, Value>
{
    Value value;
};

template <typename Types>
struct MatrixNodeData<Types, void, void>
{
};
