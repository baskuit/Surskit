#include <algorithm/algorithm.h>
#include <libpinyon/math.h>
#include <tree/tree.h>

template <IsValueModelTypes Types>
struct UCB : Types {
    using Real = typename Types::Real;

    struct MatrixStats {
        struct Data {
            int n{1};
            Types::Real v{0};
            Types::Real log_n{1};
            Types::Real q{0};
        };
        Types::template Vector<Data> row_ucb_vector;
        Types::template Vector<Data> col_ucb_vector;

        int visits{};
        PairReal<Real> value_total{0, 0};
    };
    struct ChanceStats {};
    struct Outcome {
        int row_idx, col_idx;
        Types::Value value;
    };

    class BanditAlgorithm {
       public:
        const Real c{2};

        constexpr BanditAlgorithm() {}

        constexpr BanditAlgorithm(Real c) : c{c} {}

        friend std::ostream &operator<<(std::ostream &os, const BanditAlgorithm &search) {
            os << "UCB; c: " << search.c;
            return os;
        }

        void get_empirical_strategies(const MatrixStats &stats, Types::VectorReal &row_strategy,
                                      Types::VectorReal &col_strategy) const {
            row_strategy.resize(stats.row_ucb_vector.size());
            col_strategy.resize(stats.col_ucb_vector.size());
            Real N{static_cast<Real>(stats.visits)};
            for (int row_idx{}; row_idx < stats.row_ucb_vector.size(); ++row_idx) {
                row_strategy[row_idx] = static_cast<Real>(stats.row_ucb_vector[row_idx].n) / N;
            }
            for (int col_idx{}; col_idx < stats.col_ucb_vector.size(); ++col_idx) {
                col_strategy[col_idx] = static_cast<Real>(stats.col_ucb_vector[col_idx].n) / N;
            }
        }

        void get_empirical_value(const MatrixStats &stats, Types::Value &value) const {
            const Real den = typename Types::Q{1, (stats.visits + (stats.visits == 0))};
            if constexpr (Types::Value::IS_CONSTANT_SUM) {
                value = typename Types::Value{Real{stats.value_total.get_row_value() * den}};
            } else {
                value = typename Types::Value{stats.value_total * den};
            }
        }

        void get_refined_strategies(const MatrixStats &stats, Types::VectorReal &row_strategy,
                                    Types::VectorReal &col_strategy) const {
            row_strategy.resize(stats.row_ucb_vector.size(), Real{0});
            col_strategy.resize(stats.col_ucb_vector.size(), Real{0});
            int row_idx, col_idx;
            int max_visits{0};
            for (int i{}; i < stats.row_ucb_vector.size(); ++i) {
                if (stats.row_ucb_vector[i].n > max_visits) {
                    max_visits = stats.row_ucb_vector[i].n;
                    row_idx = i;
                }
            }
            max_visits = 0;
            for (int i{}; i < stats.col_ucb_vector.size(); ++i) {
                if (stats.col_ucb_vector[i].n > max_visits) {
                    max_visits = stats.col_ucb_vector[i].n;
                    col_idx = i;
                }
            }
            row_strategy[row_idx] = Real{1};
            col_strategy[col_idx] = Real{1};
        }

        void get_refined_value(const MatrixStats &stats, Types::Value &value) const {
            get_empirical_value(stats, value);
        }

        // protected:
        void initialize_stats(int iterations, const Types::State &state, Types::Model &model,
                              MatrixStats &stats) const {}

        void expand(MatrixStats &stats, const size_t &rows, const size_t &cols,
                    const Types::ModelOutput &output) const {
            stats.row_ucb_vector.resize(rows);
            stats.col_ucb_vector.resize(cols);
        }

        void select(Types::PRNG &device, const MatrixStats &stats, Outcome &outcome) const {
            Real max_val{-1};
            for (int i{}; i < stats.row_ucb_vector.size(); ++i) {
                if (stats.row_ucb_vector[i].q > max_val) {
                    max_val = stats.row_ucb_vector[i].q;
                    outcome.row_idx = i;
                }
            }
            max_val = Real{-1};
            for (int i{}; i < stats.col_ucb_vector.size(); ++i) {
                if (stats.col_ucb_vector[i].q > max_val) {
                    max_val = stats.col_ucb_vector[i].q;
                    outcome.col_idx = i;
                }
            }
        }

        void update_matrix_stats(MatrixStats &stats, const Outcome &outcome) const {
            stats.value_total += PairReal<Real>{outcome.value.get_row_value(), outcome.value.get_col_value()};
            stats.visits += 1;
            auto &row_data = stats.row_ucb_vector[outcome.row_idx];
            auto &col_data = stats.col_ucb_vector[outcome.col_idx];

            int row_n = row_data.n;
            int col_n = col_data.n;

            row_data.v *= row_n;
            row_data.v += outcome.value.get_row_value();
            row_data.v /= (row_n += 1);

            col_data.v *= col_n;
            col_data.v += outcome.value.get_col_value();
            col_data.v /= (col_n += 1);

            row_data.log_n = std::log(row_n);
            col_data.log_n = std::log(col_n);

            Real big_log{std::log(static_cast<Real>(stats.visits))};
            if (stats.visits == 1) {
                big_log = 1;
            }

            row_data.n = row_n;
            col_data.n = col_n;

            for (auto &data : stats.row_ucb_vector) {
                data.q = data.v + big_log / data.log_n;

                for (auto &data : stats.col_ucb_vector) {
                    data.q = data.v + big_log / data.log_n;
                }
            }
        }
        void update_chance_stats(ChanceStats &stats, const Outcome &outcome) const {}
    };
};