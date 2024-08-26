#include <pinyon.h>

const size_t max_actions = 2;
const size_t max_depth = 10;
const size_t duration_ms = 5000;

template <typename Types>
void benchmark_st_()
{
    typename Types::PRNG device{0};
    typename Types::State state{max_actions, max_depth};
    typename Types::Model model{0};
    typename Types::MatrixNode root{};
    typename Types::Search search{};
    search.run(duration_ms, device, state, model, root);
    std::cout << search << " : " << root.count_matrix_nodes() << std::endl;
}

template <typename Types>
void benchmark_mt_(const size_t threads = 8)
{
    typename Types::PRNG device{0};
    typename Types::State state{max_actions, max_depth};
    typename Types::Model model{0};
    typename Types::MatrixNode root{};
    typename Types::Search search{typename Types::BanditAlgorithm{.1}, threads};
    search.run(duration_ms, device, state, model, root);
    std::cout << search << " : " << root.count_matrix_nodes() << std::endl;
}

template <typename Types>
void benchmark_mtp_(const size_t threads = 8, const size_t pool_size = 64)
{
    typename Types::PRNG device{0};
    typename Types::State state{max_actions, max_depth};
    typename Types::Model model{0};
    typename Types::MatrixNode root{};
    typename Types::Search search{typename Types::BanditAlgorithm{.1}, threads, pool_size};
    search.run(duration_ms, device, state, model, root);
    std::cout << search << " : " << root.count_matrix_nodes() << std::endl;
}

template <typename... SearchTypes>
void benchmark_st(std::tuple<SearchTypes...> search_type_tuple)
{
    (benchmark_st_<SearchTypes>(), ...);
}

template <typename... SearchTypes>
void benchmark_mt(std::tuple<SearchTypes...> search_type_tuple)
{
    (benchmark_mt_<SearchTypes>(1), ...);
    (benchmark_mt_<SearchTypes>(2), ...);
    (benchmark_mt_<SearchTypes>(4), ...);

}

template <typename... SearchTypes>
void benchmark_mtp(std::tuple<SearchTypes...> search_type_tuple)
{
    (benchmark_mtp_<SearchTypes>(1), ...);
    (benchmark_mtp_<SearchTypes>(2), ...);
    (benchmark_mtp_<SearchTypes>(4), ...);
}

int main()
{
    auto bandit_type_pack = TypePack<Exp3<MonteCarloModel<MoldState<>>>>{};
    auto node_template_pack = NodeTemplatePack<DefaultNodes, LNodes, DebugNodes, FlatNodes>{};

    auto st_search_type_tuple = search_type_generator<TreeBandit>(bandit_type_pack, node_template_pack);
    auto mt_search_type_tuple = search_type_generator<TreeBanditThreaded>(bandit_type_pack, node_template_pack);
    auto mtp_search_type_tuple = search_type_generator<TreeBanditThreadPool>(bandit_type_pack, node_template_pack);

    benchmark_st(st_search_type_tuple);
    benchmark_mt(mt_search_type_tuple);
    benchmark_mtp(mtp_search_type_tuple);
}