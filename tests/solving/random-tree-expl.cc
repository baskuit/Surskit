#include <pinyon.h>

/*

Test that tree bandit algorithms are producing low exploitability when searching on a random tree

*/

// TODO seems to be working again
// but matrix ucb refined strategies does not. Also enabling all the other shit is annoying.

// std::cout << search << std::endl;
// std::cout << "matrix:" << std::endl;
// payoff_matrix.print();
// std::cout << "strategies:" << std::endl;
// math::print(row_strategy);
// math::print(col_strategy);
// std::cout << "visits:" << std::endl;
// math::print(root.stats.row_visits);
// math::print(root.stats.col_visits);

const double expl_threshold = .3;

template <typename Types, typename SolvedState>
void test_st(const typename Types::State &state, const SolvedState &solved_state) {
    typename Types::PRNG device{0};
    typename Types::Model model{0};
    typename Types::MatrixNode root{};
    typename Types::Search search{typename Types::BanditAlgorithm{.01}};
    const size_t iterations = 1 << 12;

    search.run_for_iterations(iterations, device, state, model, root);
    typename Types::VectorReal row_strategy, col_strategy;
    typename Types::MatrixValue payoff_matrix;
    solved_state.get_matrix(payoff_matrix);
    search.get_empirical_strategies(root.stats, row_strategy, col_strategy);
    const auto expl = math::exploitability(payoff_matrix, row_strategy, col_strategy);
    search.get_refined_strategies(root.stats, row_strategy, col_strategy);
    const auto refined_expl = math::exploitability(payoff_matrix, row_strategy, col_strategy);

    std::cout << "expl: " << expl << std::endl;
    std::cout << "refined expl: " << refined_expl << std::endl;

    assert(expl < expl_threshold);
    // assert(refined_expl < expl_threshold);
}

template <typename Types, typename SolvedState>
void test_mt(const typename Types::State &state, const SolvedState &solved_state) {
    typename Types::PRNG device{0};
    typename Types::Model model{0};
    typename Types::MatrixNode root{};
    const size_t threads = 8;
    typename Types::Search search{typename Types::BanditAlgorithm{.1}, threads};
    const size_t iterations = 1 << 15;

    search.run_for_iterations(iterations, device, state, model, root);
    typename Types::VectorReal row_strategy, col_strategy;
    typename Types::MatrixValue payoff_matrix;
    solved_state.get_matrix(payoff_matrix);
    search.get_empirical_strategies(root.stats, row_strategy, col_strategy);
    auto expl = math::exploitability(payoff_matrix, row_strategy, col_strategy);
    std::cout << search << " : " << expl << std::endl;
    assert(expl < typename Types::Real{expl_threshold});
}

template <typename Types, typename SolvedState>
void test_mtp(const typename Types::State &state, const SolvedState &solved_state) {
    typename Types::PRNG device{0};
    typename Types::Model model{0};
    typename Types::MatrixNode root{};
    const size_t threads = 8;
    const size_t pool_size = 64;
    typename Types::Search search{typename Types::BanditAlgorithm{.1}, threads, pool_size};
    const size_t iterations = 1 << 15;

    search.run_for_iterations(iterations, device, state, model, root);
    typename Types::VectorReal row_strategy, col_strategy;
    typename Types::MatrixValue payoff_matrix;
    solved_state.get_matrix(payoff_matrix);
    search.get_empirical_strategies(root.stats, row_strategy, col_strategy);
    auto expl = math::exploitability(payoff_matrix, row_strategy, col_strategy);
    std::cout << search << " : " << expl << std::endl;
    assert(expl < typename Types::Real{expl_threshold});
}

template <typename... SearchTypes, typename State, typename SolvedState>
void test_expl_st(const std::tuple<SearchTypes...> search_type_tuple, const State &state,
                  const SolvedState &solved_state) {
    (test_st<SearchTypes>(state, solved_state), ...);
}

template <typename... SearchTypes, typename State, typename SolvedState>
void test_expl_mt(const std::tuple<SearchTypes...> search_type_tuple, const State &state,
                  const SolvedState &solved_state) {
    (test_mt<SearchTypes>(state, solved_state), ...);
}

template <typename... SearchTypes, typename State, typename SolvedState>
void test_expl_mtp(const std::tuple<SearchTypes...> search_type_tuple, const State &state,
                   const SolvedState &solved_state) {
    (test_mtp<SearchTypes>(state, solved_state), ...);
}

int main() {
    using BaseTypes = MonteCarloModel<RandomTree<>>;

    auto bandit_type_pack = TypePack<Exp3<BaseTypes>, MatrixUCB<BaseTypes>>{};
    // auto node_template_pack =
    //     NodeTemplatePack<
    //         DefaultNodes,
    //         LNodes,
    //         DebugNodes,
    //         FlatNodes>{};
    auto node_template_pack = NodeTemplatePack<DefaultNodes>{};
    auto st_search_type_tuple = search_type_generator<TreeBandit>(bandit_type_pack, node_template_pack);
    auto mt_search_type_tuple = search_type_generator<TreeBanditThreaded>(bandit_type_pack, node_template_pack);
    auto mtp_search_type_tuple = search_type_generator<TreeBanditThreadPool>(bandit_type_pack, node_template_pack);

    BaseTypes::Model model{0};

    RandomTreeGenerator<> generator{prng{0}, {1, 2}, {2}, {1, 3}, {0}, std::vector<size_t>(100, 0)};

    for (const auto &wrapped_state : generator) {
        const BaseTypes::State state = (wrapped_state.unwrap<BaseTypes>());
        auto solved_state = TraversedState<BaseTypes>::State{state, model};

        test_expl_st(st_search_type_tuple, state, solved_state);
        // test_expl_mt(mt_search_type_tuple, state, solved_state);
        // test_expl_mtp(mtp_search_type_tuple, state, solved_state);
        // test_expl_op(op_search_type_tuple, state, solved_state);
    }

    return 0;
}