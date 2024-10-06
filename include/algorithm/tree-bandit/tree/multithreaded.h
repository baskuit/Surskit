#pragma once

#include <algorithm/tree-bandit/tree/tree-bandit.h>

#include <tree/tree.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

template <
    IsMultithreadedBanditTypes Types,
    template <typename...> typename NodePair = DefaultNodes,
    typename Options = SearchOptions<>>
struct TreeBanditThreaded : Types
{
    struct MatrixStats : Types::MatrixStats
    {
        typename Types::Mutex stats_mutex{};
        typename Types::Mutex tree_mutex{};
    };
    struct ChanceStats : Types::ChanceStats
    {
        // typename Types::Mutex mutex{};
        // tree mutex guards both, currently
    };
    using MatrixNode = NodePair<Types, MatrixStats, ChanceStats, typename Options::NodeActions>::MatrixNode;
    using ChanceNode = NodePair<Types, MatrixStats, ChanceStats, typename Options::NodeActions>::ChanceNode;

    class Search : public Types::BanditAlgorithm
    {
    public:
        using Types::BanditAlgorithm::BanditAlgorithm;

        Search(const Types::BanditAlgorithm &base) : Types::BanditAlgorithm{base}
        {
        }

        Search(const Types::BanditAlgorithm &base, size_t threads) : Types::BanditAlgorithm{base}, threads{threads}
        {
        }

        friend std::ostream &operator<<(std::ostream &os, const Search &search)
        {
            os << "TreeBanditThreaded; threads: " << search.threads << " - ";
            os << static_cast<typename Types::BanditAlgorithm>(search);
            os << " - " << NodePair<Types, typename Types::MatrixStats, typename Types::ChanceStats>{};
            return os;
        }

        const size_t threads = 1;

        size_t run(
            const size_t duration_ms,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode &matrix_node) const
        {
            std::thread thread_pool[threads];
            size_t iterations[threads];
            size_t total_iterations = 0;
            for (int i = 0; i < threads; ++i)
            {
                thread_pool[i] = std::thread(
                    &Search::run_thread, this, duration_ms, device.uniform_64(), &state, &model, &matrix_node, std::next(iterations, i));
            }
            for (int i = 0; i < threads; ++i)
            {
                thread_pool[i].join();
            }
            for (int i = 0; i < threads; ++i)
            {
                total_iterations += iterations[i];
            }
            return total_iterations;
        }

        size_t run_for_iterations(
            const size_t iterations,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode &matrix_node) const
        {
            std::thread thread_pool[threads];
            const size_t iterations_per_thread = iterations / threads;
            const auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < threads; ++i)
            {
                thread_pool[i] = std::thread(
                    &Search::run_thread_for_iterations, this, iterations_per_thread, device.uniform_64(), &state, &model, &matrix_node);
            }
            for (int i = 0; i < threads; ++i)
            {
                thread_pool[i].join();
            }
            const auto end = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            return duration.count();
        }

        void run_thread(
            const size_t duration_ms,
            const Types::Seed thread_device_seed,
            const Types::State *state,
            const Types::Model *model,
            MatrixNode *const matrix_node,
            size_t *iterations) const
        {
            typename Types::PRNG device_thread(thread_device_seed);
            typename Types::Model model_thread{*model};
            typename Types::ModelOutput model_output;

            const auto start = std::chrono::high_resolution_clock::now();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            size_t thread_iterations = 0;
            for (; duration.count() < duration_ms; ++thread_iterations)
            {
                typename Types::State state_copy{*state};
                state_copy.randomize_transition(device_thread);
                this->run_iteration(device_thread, state_copy, model_thread, matrix_node, model_output);
                end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            }
            *iterations = thread_iterations;
        }

        void run_thread_for_iterations(
            const size_t iterations,
            const Types::Seed thread_device_seed,
            const Types::State *state,
            const Types::Model *model,
            MatrixNode *const matrix_node) const
        {
            typename Types::PRNG device_thread(thread_device_seed);
            typename Types::Model model_thread{*model};
            typename Types::ModelOutput model_output;
            for (size_t iteration = 0; iteration < iterations; ++iteration)
            {
                typename Types::State state_copy{*state};
                state_copy.randomize_transition(device_thread);
                this->run_iteration(device_thread, state_copy, model_thread, matrix_node, model_output);
            }
        }

        MatrixNode *run_iteration(
            Types::PRNG &device,
            Types::State &state,
            Types::Model &model,
            MatrixNode *const matrix_node,
            Types::ModelOutput &model_output) const
        {
            typename Types::Mutex &stats_mutex{matrix_node->stats.stats_mutex};
            typename Types::Mutex &tree_mutex{matrix_node->stats.tree_mutex};

            // no longer allows for artifically terminal nodes
            if (state.terminal())
            {
                matrix_node->set_terminal();
                model_output.value = state.get_payoff();
                return matrix_node;
            }
            else
            {
                if (!matrix_node->is_expanded())
                {
                    stats_mutex.lock();
                    if (matrix_node->is_expanded())
                    {
                        stats_mutex.unlock();
                    }
                    else
                    {
                        if constexpr (!std::is_same_v<typename Options::NodeActions, void>)
                        {
                            state.get_actions(
                                matrix_node->row_actions,
                                matrix_node->col_actions);
                            const size_t rows = matrix_node->row_actions.size();
                            const size_t cols = matrix_node->col_actions.size();
                            model.inference(std::move(state), model_output);
                            this->expand(matrix_node->stats, rows, cols, model_output);
                            stats_mutex.unlock();
                            matrix_node->expand(rows, cols);
                        }
                        else
                        {
                            const size_t rows = state.row_actions.size();
                            const size_t cols = state.col_actions.size();
                            model.inference(std::move(state), model_output);
                            this->expand(matrix_node->stats, rows, cols, model_output);
                            stats_mutex.unlock();
                            matrix_node->expand(rows, cols);
                        }
                    }
                    if constexpr (!std::is_same_v<typename Options::NodeValue, void>)
                    {
                        matrix_node->value = model_output.value;
                    }
                    if constexpr (std::is_same_v<typename Options::return_after_expand, void>)
                    {
                        return matrix_node;
                    }
                }
                else
                {
                    typename Types::Outcome outcome;
                    this->select(device, matrix_node->stats, outcome);

                    if constexpr (!std::is_same_v<typename Options::NodeActions, void>)
                    {
                        state.apply_actions(
                            matrix_node->row_actions[outcome.row_idx],
                            matrix_node->col_actions[outcome.col_idx]);
                    }
                    else
                    {
                        state.apply_actions(
                            state.row_actions[outcome.row_idx],
                            state.col_actions[outcome.col_idx]);
                        state.get_actions();
                    }

                    tree_mutex.lock();
                    ChanceNode *chance_node = matrix_node->access(outcome.row_idx, outcome.col_idx);
                    MatrixNode *matrix_node_next = chance_node->access(state.get_obs());
                    tree_mutex.unlock();

                    MatrixNode *matrix_node_leaf = run_iteration(device, state, model, matrix_node_next, model_output);

                    if constexpr (std::is_same_v<typename Options::update_using_average, void>)
                    {
                        outcome.value = model_output.value;
                    }
                    else
                    {
                        this->get_empirical_value(matrix_node_next->stats, outcome.value);
                    }
                    this->update_matrix_stats(matrix_node->stats, outcome, stats_mutex);
                    this->update_chance_stats(chance_node->stats, outcome); // no guard
                    return matrix_node_leaf;
                }
            }
        }
    };
};

template <
    IsMultithreadedBanditTypes Types,
    template <typename...> typename NodePair = DefaultNodes,
    typename Options = SearchOptions<>>
struct TreeBanditThreadPool : Types
{
    struct MatrixStats : Types::MatrixStats
    {
        int mutex_index = 0;
        std::atomic<int> atomic_mutex_index{-1};
    };
    struct ChanceStats : Types::ChanceStats
    {
    };
    using MatrixNode = NodePair<Types, MatrixStats, ChanceStats, typename Options::NodeActions>::MatrixNode;
    using ChanceNode = NodePair<Types, MatrixStats, ChanceStats, typename Options::NodeActions>::ChanceNode;

    struct DoubleMutex
    {
        DoubleMutex() {}
        DoubleMutex(const DoubleMutex &other)
        {
        }
        typename Types::Mutex first_mutex;
        typename Types::Mutex second_mutex;
    };

    class Search : public Types::BanditAlgorithm
    {
    public:
        using Types::BanditAlgorithm::BanditAlgorithm;

        Search(const Types::BanditAlgorithm &base) : Types::BanditAlgorithm{base}
        {
            mutex_pool.resize(pool_size);
        }

        Search(const Types::BanditAlgorithm &base, const size_t threads, const size_t pool_size)
            : Types::BanditAlgorithm{base}, threads{threads}, pool_size{pool_size}
        {
            mutex_pool.resize(pool_size);
        }

        Search(const Search &other) : Types::BanditAlgorithm{other}, threads{other.threads}, pool_size{other.pool_size}
        {
            mutex_pool.resize(pool_size);
        }
        // we want this class to be copyable, but atomics are not copyable
        // so we define a new copy constr that just constructs a new atomic.

        friend std::ostream &operator<<(std::ostream &os, const Search &search)
        {
            os << "TreeBanditThreadPool; threads: " << search.threads << ", pool size: " << search.pool_size << " - ";
            os << static_cast<typename Types::BanditAlgorithm>(search);
            os << " - " << NodePair<Types, typename Types::MatrixStats, typename Types::ChanceStats>{};
            return os;
        }

        const size_t threads = 1;
        const size_t pool_size = 64;
        std::vector<DoubleMutex> mutex_pool{};
        std::atomic<unsigned int> current_index{0};

        size_t run(
            const size_t duration_ms,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode &matrix_node)
        {
            std::thread thread_pool[threads];
            size_t iterations[threads];
            size_t total_iterations = 0;
            for (int i = 0; i < threads; ++i)
            {
                thread_pool[i] = std::thread(
                    &Search::run_thread, this, duration_ms, device.uniform_64(), &state, &model, &matrix_node, std::next(iterations, i));
            }
            for (int i = 0; i < threads; ++i)
            {
                thread_pool[i].join();
            }
            for (int i = 0; i < threads; ++i)
            {
                total_iterations += iterations[i];
            }
            return total_iterations;
        }

        size_t run_for_iterations(
            const size_t iterations,
            Types::PRNG &device,
            const Types::State &state,
            Types::Model &model,
            MatrixNode &matrix_node)
        {
            std::thread thread_pool[threads];
            size_t iterations_per_thread = iterations / threads;
            const auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < threads; ++i)
            {
                thread_pool[i] = std::thread(
                    &Search::run_thread_for_iterations, this, iterations_per_thread, device.uniform_64(), &state, &model, &matrix_node);
            }
            for (int i = 0; i < threads; ++i)
            {
                thread_pool[i].join();
            }
            const auto end = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            return duration.count();
        }

        void run_thread(
            const size_t duration_ms,
            const Types::Seed thread_device_seed,
            const Types::State *state,
            const Types::Model *model,
            MatrixNode *const matrix_node,
            size_t *iterations)
        {
            typename Types::PRNG device_thread{thread_device_seed}; // TODO deterministically provide new seed
            typename Types::Model model_thread{*model};             // TODO go back to not making new ones? Perhaps only device needs new instance
            typename Types::ModelOutput model_output;

            const auto start = std::chrono::high_resolution_clock::now();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            size_t thread_iterations = 0;
            for (; duration.count() < duration_ms; ++thread_iterations)
            {
                typename Types::State state_copy{*state};
                state_copy.randomize_transition(device_thread);
                this->run_iteration(device_thread, state_copy, model_thread, matrix_node, model_output);
                end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            }
            *iterations = thread_iterations;
        }

        void run_thread_for_iterations(
            const size_t iterations,
            const Types::Seed thread_device_seed,
            const Types::State *state,
            const Types::Model *model,
            MatrixNode *const matrix_node)
        {
            typename Types::PRNG device_thread{thread_device_seed};
            typename Types::Model model_thread{*model};
            typename Types::ModelOutput model_output;
            for (size_t iteration = 0; iteration < iterations; ++iteration)
            {
                typename Types::State state_copy{*state};
                state_copy.randomize_transition(device_thread);
                this->run_iteration(device_thread, state_copy, model_thread, matrix_node, model_output);
            }
        }

        MatrixNode *run_iteration(
            Types::PRNG &device,
            Types::State &state,
            Types::Model &model,
            MatrixNode *const matrix_node,
            Types::ModelOutput &model_output)
        {
            if (state.terminal())
            {
                matrix_node->set_terminal();
                model_output.value = state.get_payoff();
                return matrix_node;
            }
            else
            {
                if (!matrix_node->is_expanded())
                {
                    int expected = -1;
                    int desired = (this->current_index.fetch_add(1)) % pool_size;
                    matrix_node->stats.atomic_mutex_index.compare_exchange_weak(expected, desired);
                    if (expected != -1)
                    {
                        desired = expected;
                    }
                    else
                    {
                        matrix_node->stats.mutex_index = desired;
                    }
                    auto &mutex{this->mutex_pool[desired].first_mutex};
                    // now all nodes agree on correct mutex
                    if (mutex.try_lock())
                    {
                        if (matrix_node->is_expanded())
                        {
                            mutex.unlock();
                            return matrix_node;
                        }
                        if constexpr (!std::is_same_v<typename Options::NodeActions, void>)
                        {
                            state.get_actions(
                                matrix_node->row_actions,
                                matrix_node->col_actions);
                            const size_t rows = matrix_node->row_actions.size();
                            const size_t cols = matrix_node->col_actions.size();
                            model.inference(std::move(state), model_output);
                            this->expand(matrix_node->stats, rows, cols, model_output);
                            mutex.unlock();
                            matrix_node->expand(rows, cols);
                        }
                        else
                        {
                            const size_t rows = state.row_actions.size();
                            const size_t cols = state.col_actions.size();
                            model.inference(std::move(state), model_output);
                            this->expand(matrix_node->stats, rows, cols, model_output);
                            mutex.unlock();
                            matrix_node->expand(rows, cols);
                        }
                    }
                    if constexpr (!std::is_same_v<typename Options::NodeValue, void>)
                    {
                        matrix_node->value = model_output.value;
                    }
                    if constexpr (std::is_same_v<typename Options::return_after_expand, void>)
                    {
                        return matrix_node;
                    }
                }
                else
                {
                    auto &stats_mutex = mutex_pool[matrix_node->stats.mutex_index].first_mutex;
                    auto &tree_mutex = mutex_pool[matrix_node->stats.mutex_index].second_mutex;

                    typename Types::Outcome outcome;
                    this->select(device, matrix_node->stats, outcome);

                    if constexpr (!std::is_same_v<typename Options::NodeActions, void>)
                    {
                        state.apply_actions(
                            matrix_node->row_actions[outcome.row_idx],
                            matrix_node->col_actions[outcome.col_idx]);
                    }
                    else
                    {
                        state.apply_actions(
                            state.row_actions[outcome.row_idx],
                            state.col_actions[outcome.col_idx]);
                        state.get_actions();
                    }

                    tree_mutex.lock();
                    ChanceNode *chance_node = matrix_node->access(outcome.row_idx, outcome.col_idx);
                    MatrixNode *matrix_node_next = chance_node->access(state.get_obs());
                    tree_mutex.unlock();

                    MatrixNode *matrix_node_leaf = run_iteration(device, state, model, matrix_node_next, model_output);

                    if constexpr (std::is_same_v<typename Options::update_using_average, void>)
                    {
                        outcome.value = model_output.value;
                    }
                    else
                    {
                        this->get_empirical_value(matrix_node_next->stats, outcome.value);
                    }

                    this->update_matrix_stats(matrix_node->stats, outcome, stats_mutex);
                    this->update_chance_stats(chance_node->stats, outcome); // no guard
                    return matrix_node_leaf;
                }
            }
        }
    };
};
