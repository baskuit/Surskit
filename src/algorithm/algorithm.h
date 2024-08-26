#pragma once

#include <model/model.h>

template <typename Types>
concept IsBanditAlgorithmTypes =
    requires(
        typename Types::BanditAlgorithm &bandit,
        typename Types::PRNG &device,
        typename Types::State &state,
        typename Types::Model &model,
        typename Types::MatrixStats &matrix_stats,
        typename Types::ChanceStats &chance_stats,
        typename Types::Outcome &outcome,
        typename Types::VectorReal &strategy,
        typename Types::ModelOutput &model_output,
        typename Types::Value &value,
        typename Types::Real &real) {
        {
            bandit.get_empirical_strategies(matrix_stats, strategy, strategy)
        } -> std::same_as<void>;
        {
            bandit.get_empirical_value(matrix_stats, value)
        } -> std::same_as<void>;
        {
            bandit.get_refined_strategies(matrix_stats, strategy, strategy)
        } -> std::same_as<void>;
        {
            bandit.get_refined_value(matrix_stats, value)
        } -> std::same_as<void>;
        {
            bandit.initialize_stats(0, state, model, matrix_stats)
        } -> std::same_as<void>;
        {
            bandit.expand(matrix_stats, 0, 0, model_output)
        } -> std::same_as<void>;
        {
            bandit.select(device, matrix_stats, outcome)
        } -> std::same_as<void>;
        {
            bandit.update_matrix_stats(matrix_stats, outcome)
        } -> std::same_as<void>;
        {
            bandit.update_chance_stats(chance_stats, outcome)
        } -> std::same_as<void>;
        {
            outcome.row_idx
        } -> std::same_as<int&>;
        {
            outcome.col_idx
        } -> std::same_as<int&>;
        {
            outcome.value
        } -> std::same_as<typename Types::Value &>;
    } &&
    IsValueModelTypes<Types>;

template <typename Types>
concept IsMultithreadedBanditTypes =
    requires(
        typename Types::BanditAlgorithm &bandit,
        typename Types::PRNG &device,
        typename Types::MatrixStats &matrix_stats,
        typename Types::ChanceStats &chance_stats,
        typename Types::Outcome &outcome,
        typename Types::Mutex &mutex) {
        {
            bandit.select(device, matrix_stats, outcome, mutex)
        } -> std::same_as<void>;
        {
            bandit.update_matrix_stats(matrix_stats, outcome, mutex)
        } -> std::same_as<void>;
        {
            bandit.update_chance_stats(chance_stats, outcome, mutex)
        } -> std::same_as<void>;
    } &&
    IsBanditAlgorithmTypes<Types>;

template <typename Types>
concept IsOffPolicyBanditTypes =
    requires(
        typename Types::BanditAlgorithm &bandit,
        typename Types::PRNG &device,
        typename Types::MatrixStats &matrix_stats,
        typename Types::ChanceStats &chance_stats,
        typename Types::Outcome &outcome,
        typename Types::VectorReal &strategy,
        typename Types::Real &real) {
        {
            bandit.update_matrix_stats(matrix_stats, outcome, real)
        } -> std::same_as<void>;
        {
            bandit.update_chance_stats(chance_stats, outcome, real)
        } -> std::same_as<void>;
        {
            bandit.get_policy(matrix_stats, strategy, strategy)
        } -> std::same_as<void>;
    } &&
    IsBanditAlgorithmTypes<Types>;

template <typename Types>
concept IsTreeBanditTypes =
    requires(
        typename Types::Search &search,
        const typename Types::Search &const_search,
        typename Types::PRNG &device,
        typename Types::State &state,
        typename Types::Model &model,
        typename Types::MatrixNode &matrix_node) {
        {
            search = const_search
        } -> std::same_as<typename Types::Search &>;
        {
            const_search.run(0, device, state, model, matrix_node)
        } -> std::same_as<size_t>;
        {
            const_search.run_for_iterations(0, device, state, model, matrix_node)
        } -> std::same_as<size_t>;
    } &&
    IsBanditAlgorithmTypes<Types>;

template <
    typename rae = void, 
    typename uua = void, 
    typename node_actions = void, 
    typename node_value = void,
    size_t max_iter = 1 << 10,
    size_t max_d = 1 << 4>
struct SearchOptions
{
    // if false, iterations always rollout until terminal
    using return_after_expand = rae;
    // refer to MCTS-A. false is the faster but unproven behaviour
    using update_using_average = uua;
    // trade-off between storing actions in node vs calling get_actions() frequently
    using NodeActions = node_actions;
    // useful for algorithm agnostic pruning, but otherwise not needed
    using NodeValue = node_value;

    static const size_t max_iterations = max_iter;
    static const size_t max_depth = max_d;
};

template <typename Types>
concept IsSearchTypes =
    requires(
        typename Types::Search &search,
        typename Types::PRNG &device,
        typename Types::State &state,
        typename Types::Model &model,
        typename Types::MatrixNode &matrix_node) {
        {
            search.run(0, device, state, model, matrix_node)
        } -> std::same_as<size_t>;
    } &&
    IsValueModelTypes<Types>;
