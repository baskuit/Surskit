#include <pinyon.h>

// using UnsolvedStateTypes = OneSumMatrixGame;
using UnsolvedStateTypes = RandomTree<>;
using SolvedStateTypes = TraversedState<NullModel<UnsolvedStateTypes>>;
using FinalStateTypes = UnsolvedStateTypes;

W::Types::State generator_function(const W::Types::Seed seed)
{
    prng a_device{seed};

    const size_t depth_bound = 10;
    const size_t actions = 3;
    const size_t transitions = 1;

    UnsolvedStateTypes::State state{a_device.uniform_64(), depth_bound, actions, actions, transitions, Rational<>{0}};
    // UnsolvedStateTypes::State state{a_device.uniform_64(), actions, actions};
    // state.payoff_matrix.print();
    // UnsolvedStateTypes::State state{2, 2};
    // NullModel<UnsolvedStateTypes>::Model model{};
    // SolvedStateTypes::State solved_state{state, model};

    return W::make_state<FinalStateTypes>(state);
}

int main()
{

    using MCTypes = TreeBandit<Exp3<MonteCarloModel<FinalStateTypes>>>;
    // Exp3 search types on a solved random tree
    using MCMTypes = SearchModel<MCTypes, true, true>;
    // Model type that treats search output as its inference
    using ModelBanditTypes = TreeBanditThreaded<Exp3Fat<MonteCarloModel<ModelBandit>>>;
    // Type list for multithreaded exp3 over ModelBandit state

    std::vector<W::Types::Model> models{};
    models.emplace_back(W::make_model<NullModel<FinalStateTypes>>());
    const size_t monte_carlo_iterations = 1 << 10;
    models.emplace_back(W::make_model<MCMTypes>(MCMTypes::Model{monte_carlo_iterations, {0}, {0}, {.7}}));
    models.emplace_back(W::make_model<MCMTypes>(MCMTypes::Model{monte_carlo_iterations, {0}, {0}, {.1}}));
    models.emplace_back(W::make_model<MCMTypes>(MCMTypes::Model{monte_carlo_iterations, {0}, {0}, {.15}}));
    // models.emplace_back(W::make_model<SolvedStateModel<SolvedStateTypes>>());

    W::Types::ModelOutput output_;
    models[1].inference(generator_function(0), output_);
    math::print(output_.row_policy);
    math::print(output_.col_policy);

    const size_t threads = 4;
    const size_t vs_rounds = 1;
    ModelBanditTypes::PRNG device{0};
    ModelBanditTypes::State arena_state{&generator_function, models, vs_rounds};
    ModelBanditTypes::Model arena_model{1337};
    ModelBanditTypes::Search search{ModelBanditTypes::BanditAlgorithm{.10}, threads};
    ModelBanditTypes::MatrixNode node;

    const size_t arena_search_iterations = 1 << 16;
    search.run_for_iterations(arena_search_iterations, device, arena_state, arena_model, node);

    node.stats.matrix.print();
    return 0;
}
