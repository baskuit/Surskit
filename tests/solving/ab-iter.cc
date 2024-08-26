#include <chrono>
#include <pinyon.h>

/*
Takes model types (requires the model be 'state-invariant')
and asserts that iterative deepening produces the same alpha/beta
when solving on the same state and same depth

We would also like to assert that the search requires less nodes
but that is simply not going to be the case all the time

*/
using T = RandomTree<RandomTreeRationalTypes>;

struct BiasModel : T {
    struct ModelOutput {
        T::Value value;
    };

    class Model {
       public:
        const int n{};

        Model() {}

        Model(int n) : n{n} {}

        void inference(T::State &&state, ModelOutput &output) const {
            if (state.payoff_bias < n) {
                output.value = T::Value{mpq_class{0}};
                return;
            }
            if (state.payoff_bias > n) {
                output.value = T::Value{mpq_class{1}};
                return;
            }
            output.value = T::Value{mpq_class{1, 2}};
        }
    };
};

// struct ABIterTestData {
//     size_t time_us;
//     size_t iter_time_us;
//     size_t count;
//     size_t iter_count;
// };

// template <IsValueModelTypes Types>
// struct IterTest {

//     using Flat = AlphaBetaForce<Types>;
//     using Iter = AlpbaBetaIter<Types>

//     ABIterTestData test (const BiasModel::State &) const {
//         size_t time_us, iter_time_us, count, count_iter;

//         return {time_us, iter_time_us, count, count_iter};
//     }

// };

using AB = AlphaBeta<BiasModel>;
using ABIter = AlphaBetaDev<BiasModel>;

std::pair<double, double> solve_check(const BiasModel::State &state, const size_t transitions) {
    assert(state.transitions == 1);

    using time_t = decltype(std::chrono::high_resolution_clock::now());
    time_t start, end;
    size_t us;

    const size_t max_depth = state.depth_bound;

    BiasModel::PRNG device{0};
    BiasModel::Model model{0};

    AB::Search search{};
    ABIter::Search search_iter{1, 1, 0.0};

    AB::MatrixNode node{};
    ABIter::MatrixNode node_iter{};

    start = std::chrono::high_resolution_clock::now();
    const auto pair = search.run(max_depth, device, state, model, node);

    end = std::chrono::high_resolution_clock::now();
    const size_t time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // const size_t time_iter = search_iter.run(max_depth, device, state, model, node_iter);
    // below uses vector input for depths
    const size_t time_iter = search_iter.run({max_depth/2, max_depth}, device, state, model, node_iter);

    const size_t node_count = node.count_matrix_nodes();
    const size_t node_iter_count = node_iter.count_matrix_nodes();
    std::cout << "AB: " << node_count << std::endl;
    std::cout << "AB Iter: " << node_iter_count << std::endl;
    std::cout << "AB (time): " << time << std::endl;
    std::cout << "AB Iter (time): " << time_iter << std::endl;

    assert(pair.first >= node_iter.alpha);
    assert(pair.second <= node_iter.beta);
    assert(node_iter.alpha != 0);
    assert(node_iter.beta != 1);

    if (transitions == 1) {
        assert(pair.first == node_iter.alpha);
        assert(pair.second == node_iter.beta);
    }

    return {node_iter_count / (double) node_count, time_iter / (double) time};
}

int main() {
    BiasModel::Q threshold{0};
    const size_t transitions = 1;
    RandomTreeGenerator<RandomTreeRationalTypes> generator{prng{2}, {5}, {4}, {transitions}, {threshold}, std::vector<size_t>(20, 0)};

    double total_count_ratio = 0;
    double total_time_ratio = 0;

    int tries = 0;

    size_t counter = 0;
    for (auto wrapped_state : generator) {
        const auto state = wrapped_state.unwrap<T>();
        const auto pair = solve_check(state, transitions);
        total_count_ratio += pair.first;
        total_time_ratio += pair.second;
        ++counter;
    }
    const double count_ratio = total_count_ratio / counter;
    const double time_ratio = total_time_ratio / counter;

    std::cout << "count ratio: " << count_ratio << std::endl;
    std::cout << "time_ratio: " << time_ratio << std::endl;

    return 0;
}