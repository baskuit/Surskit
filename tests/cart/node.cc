#include <pinyon.h>

/*

This test runs single-threaded and the 2 main multi-threaded implementations 
of a simple Exp3 search on MoldState.

The purpose is to check that thread contention does not result in a wildly different tree topology

TODO: Currently no failure condition besides looking wrong (lol)
Also a bit primitive by my current standards

*/

int main()
{

    using BaseTypes = MonteCarloModel<MoldState<>>;

    using SearchTypes0 = TreeBandit<Exp3<BaseTypes>>;
    using SearchTypes1 = TreeBanditThreaded<Exp3<BaseTypes>>;
    using SearchTypes2 = TreeBanditThreadPool<Exp3<BaseTypes>>;

    std::tuple<
        SearchTypes0,
        SearchTypes1,
        SearchTypes2>
        search_type_tuple{{}, {}, {}};

    std::tuple<
        SearchTypes0::Search,
        SearchTypes1::Search,
        SearchTypes2::Search>
        search_tuple{{}, {{}, 2}, {{}, 2, 64}};

    auto type_search_zipped = zip(search_type_tuple, search_tuple);

    size_t lambda_idx = 0;

    auto lambda = [&lambda_idx](auto type_search_pair)
    {
        using Types = decltype(type_search_pair.first);
        using MatrixNode = typename Types::MatrixNode;
        MatrixNode root{};

        BaseTypes::State state{2, 50};
        BaseTypes::Model model{0};

        const size_t iterations = 1 << 18;
        prng device{0};
        auto &search = type_search_pair.second;

        search.run_for_iterations(iterations, device, state, model, root);

        size_t count = root.count_matrix_nodes();

        double ratio = count / (double)iterations;

        std::cout << ratio << ", diff =" << std::abs(static_cast<double>(count) - static_cast<double>(iterations)) << std::endl;
        ++lambda_idx;
    };

    for (int i = 0; i < 100; ++i)
    {
        // Apply the lambda to all elements of the tuple using std::apply
        std::apply([&lambda](auto &...elements)
                   { (lambda(elements), ...); },
                   type_search_zipped);
    }

    return 0;
}