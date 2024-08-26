#include <pinyon.h>

template <typename Types>
void check_flat_search_equivalence(
    const size_t iterations,
    const typename Types::PRNG &device,
    const typename Types::State &state,
    const typename Types::Model &model,
    const typename Types::BanditAlgorithm &bandit)
{
    typename Types::MatrixStats stats, flat_stats;
    {
        auto device_ = device;
        auto model_ = model;
        typename TreeBandit<Types>::MatrixNode node{};
        typename TreeBandit<Types>::Search search{bandit};
        const size_t count = search.run_for_iterations(iterations, device_, state, model_, node);
        stats = node.stats;
    };
    std::cout << std::endl;
    {
        auto device_ = device;
        auto model_ = model;
        typename TreeBanditFlat<Types>::Search search{bandit};
        const size_t count = search.run_for_iterations(iterations, device_, state, model_);
        flat_stats = search.matrix_data[0].stats;
    };
    assert(stats == flat_stats);
}

template <typename Types>
void test_equiv_(
    const typename Types::State &state)
{
    const size_t iterations = 1 << 15;
    typename Types::PRNG device{0};
    typename Types::Model model{0};
    typename Types::BanditAlgorithm bandit{};

    check_flat_search_equivalence<Types>(
        iterations,
        device,
        state,
        model,
        bandit
    );
}

template <typename... BanditTypes, typename State>
void test_equiv(
    const TypePack<BanditTypes...> bandit_type_pack,
    const State &state)
{
    (test_equiv_<BanditTypes>(state), ...);
}

int main()
{
    using Types = MonteCarloModel<RandomTree<>>;

    auto bandit_type_pack =
        TypePack<
            Exp3<Types>>{};

    prng device{0};
    Types::Model model(device);

    RandomTreeGenerator<> generator{
        prng{0},
        {20},
        {3},
        {1, 2},
        {Rational<>{0, 1}, Rational<>{1, 2}},
        std::vector<size_t>(100, 0)};

    for (const auto &wrapped_state : generator)
    {
        const Types::State state = (wrapped_state.unwrap<Types>());
        std::cout << "state seed: " << state.device.get_seed() << std::endl;
        test_equiv(bandit_type_pack, state);
    }

    return 0;
}
