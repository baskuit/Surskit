#pragma once

#include <libpinyon/dynamic-wrappers.h>
#include <libpinyon/generator.h>
#include <state/state.h>
#include <types/random.h>
#include <types/types.h>
#include <vector>

template <IsTypeList Types = RandomTreeFloatTypes>
struct RandomTree : Types {
    class State : public PerfectInfoState<Types> {
       public:
        Types::PRNG device{};
        Types::Seed transition_seed{};
        size_t depth_bound = 0;
        size_t rows = 0;
        size_t cols = 0;
        size_t transitions = 1;
        int payoff_bias = 0;
        typename Types::Q chance_threshold{1, static_cast<int>(transitions + 1)};
        std::vector<typename Types::Prob> chance_strategies;
        int chance_denominator = 10;

        int (*depth_bound_func)(State *, int) = &(State::depth_bound_default);
        int (*actions_func)(State *, int) = &(State::actions_default);
        int (*payoff_bias_func)(State *, int) = &(State::payoff_bias_default);

        // everything above determines the abstract game tree exactly

        State(const Types::PRNG &device, size_t depth_bound, size_t rows, size_t cols, size_t transitions,
              const Types::Q &chance_threshold = typename Types::Q{0})
            : device{device},
              depth_bound{depth_bound},
              rows{rows},
              cols{cols},
              transitions{transitions},
              chance_threshold{chance_threshold} {
            this->init_range_actions(rows, cols);
            get_chance_strategies();
        }

        State(const Types::PRNG &device, size_t depth_bound, size_t rows, size_t cols, size_t transitions,
              Types::Q chance_threshold, int (*depth_bound_func)(State *, int), int (*actions_func)(State *, int),
              int (*payoff_bias_func)(State *, int))
            : device{device},
              depth_bound{depth_bound},
              rows{rows},
              cols{cols},
              transitions{transitions},
              chance_threshold{chance_threshold},
              depth_bound_func{depth_bound_func},
              actions_func{actions_func},
              payoff_bias_func{payoff_bias_func} {
            this->init_range_actions(rows, cols);
            get_chance_strategies();
        }

        void randomize_transition(Types::PRNG &device) { transition_seed = device.uniform_64(); }

        void randomize_transition(Types::Seed seed) { transition_seed = seed; }

        void get_actions() { this->init_range_actions(rows, cols); }

        void get_actions(Types::VectorAction &row_actions, Types::VectorAction &col_actions) const {
            row_actions.resize(rows);
            col_actions.resize(cols);
            for (int row_idx = 0; row_idx < rows; ++row_idx) {
                row_actions[row_idx] = row_idx;
            };
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                col_actions[col_idx] = col_idx;
            };
        }

        void get_chance_actions(const Types::Action row_action, const Types::Action col_action,
                                std::vector<typename Types::Obs> &chance_actions) const {
            chance_actions.clear();
            const size_t start_idx = get_transition_idx(row_action, col_action, {0});
            for (int chance_idx = 0; chance_idx < transitions; ++chance_idx) {
                if (chance_strategies[start_idx + chance_idx] > typename Types::Prob{0}) {
                    chance_actions.push_back(typename Types::Obs{chance_idx});
                }
            }
        }

        void apply_actions(Types::Action row_action, Types::Action col_action, Types::Obs chance_action) {
            const int transition_idx = get_transition_idx(row_action, col_action, chance_action);
            device.discard(transition_idx);
            // advance the Types::PRNG so that different player/chance actions have different outcomes

            this->obs = chance_action;
            this->prob = chance_strategies[transition_idx];

            depth_bound = (*depth_bound_func)(this, depth_bound);
            depth_bound *= depth_bound >= 0;
            payoff_bias = (*payoff_bias_func)(this, payoff_bias);

            if (depth_bound == 0) {
                this->terminal = true;
                typename Types::Q row_payoff{(payoff_bias > 0) - (payoff_bias < 0) + 1, 2};
                row_payoff.canonicalize();
                this->payoff = typename Types::Value{typename Types::Real{row_payoff}};
            } else {
                rows = (*actions_func)(this, rows);
                cols = (*actions_func)(this, cols);
                get_chance_strategies();
            }
        }

        const Types::Prob & get_prob () const {
            return this->prob;
        }

        void apply_actions(Types::Action row_action, Types::Action col_action) {
            // get_chance_actions has always 'just been called' since its in the ctor and apply_actions()
            // Therefore we should just be able to sample it
            typename Types::PRNG temp_device{this->transition_seed};

            const int start = this->get_transition_idx(row_action, col_action, 0);
            double p = temp_device.uniform();
            for (int c{}; c < this->transitions; ++c) {
                const double q = math::to_double(this->chance_strategies[start + c]);
                p -= q;
                // if (row_action == 1 && col_action == 1) {
                //     std::cout << '!' << c << ' ' << p << ' ' << q << std::endl;
                // }

                if (p <= 0) {
                    this->apply_actions(row_action, col_action, c);
                    return;
                }
            }
            this->apply_actions(row_action, col_action, 0);
        }

        /*
        Default Growth Functions
        */

        static int depth_bound_default(State *state, int depth) { return (depth - 1) * (depth >= 0); }
        static int actions_default(State *state, int n_actions) { return n_actions; }
        static int payoff_bias_default(State *state, int payoff_bias) {
            const int bias = 1;
            return payoff_bias + state->device.random_int(2 * bias + 1) - bias;
        }

       private:
        inline size_t get_transition_idx(Types::Action row_action, Types::Action col_action,
                                         Types::Obs chance_action) const {
            return row_action * cols * transitions + col_action * transitions + chance_action;
        }

        void get_chance_strategies() {
            std::vector<typename Types::Q> chance_strategies_;
            // place holder that uses Q rationals, because they are better behaved than mpq_class Prob's
            // I had trouble with that when fixing this function last time
            chance_strategies_.resize(rows * cols * transitions);
            this->chance_strategies.resize(rows * cols * transitions);
            for (int row_idx = 0; row_idx < rows; ++row_idx) {
                for (int col_idx = 0; col_idx < cols; ++col_idx) {
                    int start_idx = row_idx * cols * transitions + col_idx * transitions;

                    // get unnormalized distro
                    typename Types::Q prob_sum{0};
                    // typename Types::Q decay{chance_denominator - 1, chance_denominator};

                    for (int chance_idx = 0; chance_idx < transitions; ++chance_idx) {
                        const int num = device.random_int(chance_denominator) + 1;

                        typename Types::Q x{num, chance_denominator};
                        // x = x * decay;
                        // x.canonicalize();
                        if (x < chance_threshold) {
                            x = 0;
                        }
                        chance_strategies_[start_idx + chance_idx] = x;
                        prob_sum += x;

                        // decay = decay * decay;
                        // decay.canonicalize();
                    }

                    prob_sum.canonicalize();

                    if (prob_sum == typename Types::Q{0}) {
                        chance_strategies_[start_idx] = 1;
                        prob_sum = 1;
                    }

                    for (int chance_idx = 0; chance_idx < transitions; ++chance_idx) {
                        auto &x = chance_strategies_[start_idx + chance_idx];
                        x = x / prob_sum;
                        x.canonicalize();
                        this->chance_strategies[start_idx + chance_idx] = typename Types::Prob{x};
                    }
                }
            }
        }
    };
};

/*

Helper class to generate random tree instances for testing

*/

template <typename TypeList = RandomTreeFloatTypes>
struct RandomTreeGenerator
    : CartesianProductGenerator<W::Types::State, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>,
                                std::vector<Rational<>>, std::vector<size_t>> {
    inline static prng device{0};  // static because used in static member function, TODO
    // This class is not used for model-bandit, maybe remove?

    // static otherwise implcit this arg messes up signature
    static W::Types::State constr(std::tuple<size_t, size_t, size_t, Rational<>, size_t> tuple) {
        return W::Types::State{RandomTree<TypeList>{}, RandomTreeGenerator::device.uniform_64(),
                               std::get<0>(tuple),     std::get<1>(tuple),
                               std::get<1>(tuple),     std::get<2>(tuple),
                               std::get<3>(tuple)};
    };

    RandomTreeGenerator(const prng &device, const std::vector<size_t> &depth_bound_vec,
                        const std::vector<size_t> &actions_vec, const std::vector<size_t> &chance_action_vec,
                        const std::vector<Rational<>> &chance_threshold_vec, const std::vector<size_t> &trial_vec)
        : CartesianProductGenerator<W::Types::State, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>,
                                    std::vector<Rational<>>, std::vector<size_t>>{
              RandomTreeGenerator::constr, depth_bound_vec, actions_vec, chance_action_vec, chance_threshold_vec, trial_vec} {
        RandomTreeGenerator::device = prng{device};
    }
};
