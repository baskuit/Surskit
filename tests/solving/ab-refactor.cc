#include <pinyon.h>

int main() {
    constexpr bool debug = false;
    using T = MonteCarloModel<RandomTree<RandomTreeRationalTypes>>;
    using Types = AlphaBetaRefactor<T, debug>;
    using SlowTypes = AlphaBeta<T>;
    using Solver = FullTraversal<T>;

    Types::PRNG device{2341343};

    const size_t tree_depth = 4;
    const size_t transitions = 2;
    const size_t min_tries = 1;
    const size_t max_tries = 16;

    const size_t max_trials = 30;
    for (int actions = 3; actions <= 3; ++actions) {

        double total_time_ratio{};
        double total_count_ratio{};

        for (int trials = 0; trials < max_trials; ++trials) {
            const uint64_t state_seed = device.uniform_64();
            const uint64_t model_seed = device.uniform_64();
            prng local_device{device.uniform_64()};

            const Types::State state{prng{state_seed}, tree_depth, actions, actions, transitions};
            Types::Model model{prng{model_seed}};
            Types::Search search{min_tries, max_tries};
            Types::MatrixNode node{};

            Solver::Search solver{};
            Solver::MatrixNode solver_node{};

            SlowTypes::Search search_slow{};
            SlowTypes::MatrixNode node_slow{};
                const auto start = std::chrono::high_resolution_clock::now();

            const auto [a, b] = search_slow.run(tree_depth, local_device, state, model, node_slow);
                const auto end = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            size_t slow_time{duration.count()};
            std::cout << "SEED: " << state_seed << ' ' << model_seed << std::endl;

            // solver.run(tree_depth, local_device, state, model, solver_node);
            std::vector<size_t> vec{};
            vec.push_back(tree_depth);
            auto output = search.run(tree_depth, local_device, state, model, node);

            // if constexpr (debug) {

            //     solver_node.stats.nash_payoff_matrix.print();
            //     std::cout << "strategies:" << std::endl;
            //     Types::mpq_vector_print(solver_node.stats.row_solution);
            //     Types::mpq_vector_print(solver_node.stats.col_solution);
            // }
            std::cout << "alpha: " << output.alpha.get_d() << " beta: " << output.beta.get_d() << std::endl;
            // std::cout << solver_node.stats.payoff.get_row_value().get_d() << std::endl;
            std::cout << "slow alpha: " << a.get_d() << " slow beta: " << b.get_d() << std::endl;

            // std::cout << "full: " << solver_node.count_matrix_nodes() << std::endl;
            std::cout << "ab counts ";
            for (const auto c : output.counts) {
                std::cout << c << ", ";
            }
            std::cout << std::endl;
            size_t total_time = 0;
            std::cout << "ab times: ";
            for (const auto t : output.times) {
                std::cout << t << ", ";
                total_time += t;
            }
            std::cout << "= " << total_time << std::endl;
            std::cout << "slow ab time: " << slow_time << std::endl; 

            mpq_class x = solver_node.stats.payoff.get_row_value();
            math::canonicalize(x);
            assert(x >= output.alpha);
            assert(x <= output.beta);

            total_time_ratio += (double)total_time/slow_time;
            total_count_ratio += (double)output.counts.back()/node_slow.count_matrix_nodes();
        }

        std::cout << total_time_ratio / max_trials << std::endl;
        std::cout << total_count_ratio / max_trials << std::endl;

    }
}