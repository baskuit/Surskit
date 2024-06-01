#pragma once

#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <gmpxx.h>

template <typename Types>
struct AlphaBetaRefactor {

    using PRNG = typename Types::PRNG;
    using State = typename Types::State;
    using Obs = typename Types::Obs;
    using Model = typename Types::Model;
    using ModelOutput = typename Types::ModelOutput;

    static uint64_t hash(const Obs &obs) {
        return static_cast<uint64_t>(obs);
    }

    struct MatrixNode;

    struct BaseData {
        uint32_t max_depth;
        PRNG *device;
        Model *model;
        const State state;
        uint32_t min_tries;
        uint32_t max_tries;
        double max_unexplored;
        double min_chance_prob;

        BaseData(
            uint32_t max_depth,
            PRNG *device,
            Model *model,
            const State state,
            uint32_t min_tries,
            uint32_t max_tries,
            double max_unexplored,
            double min_chance_prob)
            : max_depth{max_depth}, device{device}, model{model}, state{state}, min_tries{min_tries}, max_tries{max_tries}, max_unexplored{max_unexplored}, min_chance_prob{min_chance_prob} {}
    };

    struct HeadData {
        uint32_t min_tries;
        uint32_t max_tries;
        double max_unexplored;
        double min_chance_prob;
        uint32_t depth{};

        HeadData(
            uint32_t min_tries,
            uint32_t max_tries,
            double max_unexplored,
            double min_chance_prob) : min_tries{min_tries}, max_tries{max_tries}, max_unexplored{max_unexplored}, min_chance_prob{min_chance_prob} {}

        void step_forward() {
            ++depth;
        }
        void step_back() {
            --depth;
        }
    };

    struct TempData {
        uint16_t rows, cols;
        State state{prng{1}, 1, 2, 2, 1};
        mpq_class alpha{0};
        mpq_class beta{1};
        mpq_class min_branch_prob{};

        struct ChanceStats {
            mpq_class alpha_explored{0};
            mpq_class beta_explored{0};
            mpq_class unexplored{1};
        };
        std::vector<ChanceStats> chance_stat_matrix{};

        std::vector<mpq_class> alpha_matrix{};
        std::vector<mpq_class> beta_matrix{};
        std::vector<mpq_class> row_strategy{};
        std::vector<mpq_class> col_strategy{};

        bool is_solved_exactly{true};
        bool must_break{false};
        bool new_row_action{false};
        bool new_col_action{false};

        void reset_flags_for_alpha_beta_loop() {
            must_break = true;
            new_row_action = false;
            new_col_action = false;
        }
    };

    struct Branch {

        std::unique_ptr<MatrixNode> matrix_node;
        float p;
        uint64_t seed;

        Branch(const double prob, const uint64_t seed, const bool is_terminal)
            : matrix_node{}, p{static_cast<float>(prob)}, seed{seed} {
            if (!is_terminal) {
                matrix_node = std::make_unique<MatrixNode>();
            }
        }
    };

    struct ChanceNode {
        std::unordered_map<uint64_t, Branch> branches{};
        std::vector<Branch *> sorted_branches{};
        int branch_index{};
        uint32_t tries{};

        Branch *
        try_get_new_branch(const BaseData &base_data, const HeadData &head_data, const TempData &temp_data, TempData &next_temp_data,
                           const uint8_t row_idx, const uint8_t col_idx) {

            const auto go = [&](const uint64_t seed) {
                next_temp_data.state = temp_data.state;
                next_temp_data.state.randomize_transition(seed);
                next_temp_data.state.apply_actions(
                    next_temp_data.state.row_actions[row_idx],
                    next_temp_data.state.col_actions[col_idx]);
            };

            if (branch_index < sorted_branches.size()) {
                go(sorted_branches[branch_index]->seed);
                return sorted_branches[branch_index++];
            }

            auto &chance_data = temp_data.chance_stat_matrix[row_idx * temp_data.cols + col_idx];
            while (
                !(tries > base_data.max_tries ||
                  (tries > head_data.min_tries && chance_data.unexplored < head_data.max_unexplored) ||
                  chance_data.unexplored <= 0)) {

                const uint64_t seed = base_data.device->uniform_64();
                go(seed);

                Obs obs = next_temp_data.state.get_obs();
                const uint64_t h = hash(obs);
                if (branches.find(h) == branches.end()) {
                    branches.try_emplace(
                        h,
                        next_temp_data.state.get_prob().get_d(), seed, next_temp_data.state.is_terminal());
                    Branch *new_branch = &branches.at(seed);
                    return new_branch;
                }
            }

            return nullptr;
        }

        void total_reset() {
            branches.clear();
            sorted_branches.clear();
            branch_index = 0;
            tries = 0;
        }
    };

    // std::vector<ChanceNode> complains about copy contructor, presumably because of std::move
    struct ChanceNodeMatrix {
        uint8_t rows, cols;
        ChanceNode *data;

        void init(const uint8_t rows, const uint8_t cols) {
            this->rows = rows;
            this->cols = cols;
            data = new ChanceNode[rows * cols];
        }

        ChanceNode &operator()(uint8_t row_idx, uint8_t col_idx) {
            return data[row_idx * cols + col_idx];
        }

        ~ChanceNodeMatrix() {
            if (rows | cols) {
                delete data;
            }
        }
    };

    struct MatrixNode {

        ChanceNodeMatrix chance_node_matrix;

        struct ActionProb {
            uint8_t idx;
            uint8_t discrete_prob{};

            ActionProb(const uint8_t index) : idx{index} {}

            operator uint8_t() const {
                return idx;
            }
            bool operator<(const ActionProb &a) const {
                return discrete_prob < a.discrete_prob;
            }
        };

        struct Solution {
            std::vector<ActionProb> action_indices{};
            uint8_t boundary{0};

            Solution() {}

            Solution(uint8_t size) {
                for (uint8_t i = 0; i < size; ++i) {
                    action_indices.push_back(i);
                }
            }

            void add_index(uint8_t index) {
                assert(boundary >= 0 && boundary < action_indices.size() - 1);
                std::swap(action_indices[index], action_indices[boundary++]);
            }

            void remove_index(uint8_t index) {
                assert(boundary >= 1 && boundary < action_indices.size());
                std::swap(action_indices[index], action_indices[--boundary]);
            }
        };
        Solution I;
        Solution J;

        void prune() {
            for (uint8_t i = I.boundary; i < chance_node_matrix.rows; ++i) {
                for (uint8_t j = J.boundary; j < chance_node_matrix.cols; ++j) {
                    auto &chance_node = chance_node_matrix(I.action_indices[i], J.action_indices[j]);
                    chance_node.total_reset();
                }
            }
        }
    };

    struct Search {

        void next_solve(
            TempData::ChanceStats &stats,
            MatrixNode *next_matrix_node, BaseData &base_data, HeadData &head_data, TempData &temp_data, TempData &next_temp_data) const {

            const mpq_class prob = next_temp_data.state.get_prob();
            stats.unexplored -= prob;
            mpq_class alpha;
            mpq_class beta;
            if (next_temp_data.state.is_terminal()) {
                alpha = beta = next_temp_data.state.get_payoff().get_row_value();
            } else if (head_data.min_chance_prob > prob) { // TODO init min_chance_prob
                alpha = 0;
                beta = 1;
            } else if (head_data.depth + 1 == base_data.max_depth) {
                ModelOutput output{};
                base_data.model->inference(std::move(next_temp_data.state), output);
                alpha = beta = output.value.get_row_value();
            } else {
                this->alpha_beta(next_matrix_node, base_data, head_data, next_temp_data);
                alpha = next_temp_data.alpha;
                beta = next_temp_data.beta;
            }
            stats.alpha_explored += alpha * prob;
            stats.beta_explored += beta * prob;
            temp_data.is_solved_exactly &= (alpha == beta);
        }

        void solve_chance_node(
            MatrixNode *matrix_node, BaseData &base_data, HeadData &head_data, TempData &temp_data, TempData &next_temp_data,
            const uint8_t row_idx, const uint8_t col_idx) const {

            auto &chance_data = temp_data.chance_stat_matrix[0];
            Branch *new_branch;
            ChanceNode &chance_node = matrix_node->chance_node_matrix(row_idx, col_idx);
            while (new_branch = chance_node.try_get_new_branch(base_data, head_data, temp_data, next_temp_data, row_idx, col_idx)) {
                next_solve(chance_data, new_branch->matrix_node.get(), base_data, head_data, temp_data, next_temp_data);
            }
        }

        std::pair<mpq_class, mpq_class>
        tentatively_solve_subgame(
            MatrixNode *matrix_node, BaseData &base_data, HeadData &head_data, TempData &temp_data, TempData &next_temp_data) const {
            if (temp_data.is_solved_exactly) {

            } else {
            }
            return {};
        }

        void initialize_submatrix(
            MatrixNode *matrix_node, BaseData &base_data, HeadData &head_data, TempData &temp_data, TempData &next_temp_data) const {

            if (!matrix_node->chance_node_matrix.rows) {
                matrix_node->chance_node_matrix.init(temp_data.state.row_actions.size(), temp_data.state.col_actions.size());
            }

            // prune here possibly

            for (uint8_t i = 0; i < matrix_node->I.boundary; ++i) {
                const uint8_t row_idx = matrix_node->I.action_indices[i].idx;
                for (uint8_t j = 0; j < matrix_node->J.boundary; ++j) {
                    const uint8_t col_idx = matrix_node->J.action_indices[j].idx;
                    solve_chance_node(matrix_node, base_data, head_data, temp_data, next_temp_data, row_idx, col_idx);
                }
            }
        }

        // pass a reference to best score
        void row_modify_beta_and_add_action(
            mpq_class &best_response,
            MatrixNode *matrix_node, BaseData &base_data, HeadData &head_data, TempData &temp_data, TempData &next_temp_data) const {

            bool added_new_action = false;
            uint8_t new_row_idx;

            struct BestResponse {
                const uint8_t row_idx;
                uint8_t col_idx;
                mpq_class value{};
                mpq_class total_unexplored{};
                std::vector<mpq_class> priority_vector{};
            };

            const auto find_nonzero_priority = [&](uint8_t &max_index, const BestResponse &data) {
                mpq_class max_prio{-1};
                bool nonzero_piority_found{false};
                for (uint8_t j = 0; j < matrix_node->J.boundary; ++j) {
                    if (data.priority_vector[j] > std::max(mpq_class{}, max_prio)) {
                        max_index = j;
                        max_prio = data.priority_vector[j];
                        nonzero_piority_found = true;
                    }
                }
                return nonzero_piority_found;
            };

            const auto can_beat_best_response = [&](BestResponse &data) {
                return data.value + data.total_unexplored > best_response;
            };

            const auto compute_value = [&](const uint8_t i, BestResponse &data) {
                data.priority_vector.resize(matrix_node->J.boundary);

                // set up priority, score vector
                {
                    for (uint8_t j = 0; j < matrix_node->J.boundary; ++j) {
                        if (matrix_node->J.action_indices[j].discrete_prob == 0) {
                            continue;
                        }
                        data.col_idx = matrix_node->J.action_indices[j].idx;
                        const uint8_t chance_stat_index = data.row_idx * temp_data.cols + data.col_idx;

                        data.priority_vector[j] =
                            temp_data.chance_stat_matrix[chance_stat_index].unexplored *
                            temp_data.col_strategy[j];
                        data.total_unexplored += data.priority_vector[j];
                    }
                }

                // explore as much as possible
                for (uint8_t j; can_beat_best_response(data) && find_nonzero_priority(j, data);) {
                    ChanceNode &chance_node = matrix_node->chance_node_matrix(data.row_idx, j);
                    Branch *branch = chance_node.try_get_new_branch(base_data, head_data, temp_data, next_temp_data, data.row_idx, data.col_idx);
                    if (branch == nullptr) {
                        data.priority_vector[j] = 0;
                    } else [[likely]] {
                        const mpq_class prob = next_temp_data.state.get_prob();
                        const uint8_t entry_idx = data.row_idx * temp_data.cols + data.col_idx;
                        next_solve(temp_data.chance_stat_matrix[entry_idx], branch->matrix_node.get(), base_data, head_data, temp_data, next_temp_data);
                        data.priority_vector[j] -= prob * temp_data.col_strategy[data.col_idx]; // TODO maybe add prob to temp_data
                        data.value += next_temp_data.beta * prob * temp_data.col_strategy[data.col_idx];
                    }
                }

                data.value += data.total_unexplored;
                data.value.canonicalize();

                // TODO i think we need >= here because failing to add new, equally scoring best responses won't solve the game!
                if (data.value >= best_response) {
                    added_new_action = true;
                    new_row_idx = data.row_idx;
                    best_response = std::move(data.value);
                }
            };

            // actions in I
            for (uint8_t i = 0; i < matrix_node->I.boundary; ++i) {
            }

            // actions not in I
            for (uint8_t i = matrix_node->I.boundary; i < matrix_node->chance_node_matrix.rows; ++i) {
                BestResponse data{matrix_node->I.action_indices[i].idx};
                compute_value(i, data);
            }

            // solve chance nodes for final, new action
            if (added_new_action) {
                for (uint8_t col_idx = 0; col_idx < temp_data.cols; ++col_idx) {
                    this->solve_chance_node(matrix_node, base_data, head_data, temp_data, next_temp_data, new_row_idx, col_idx);
                }
                temp_data.must_break = false;
            }

            temp_data.new_row_action = added_new_action;
        }

        void col_modify_beta_and_add_action(
            mpq_class &alpha,
            MatrixNode *matrix_node, BaseData &base_data, HeadData &head_data, TempData &temp_data, TempData &next_temp_data) const {

            struct BestResponse {
            };
        }

        void sort_branches_and_reset(MatrixNode *matrix_node) const {
            for (uint8_t i = 0; i < matrix_node->chance_node_matrix.rows * matrix_node->chance_node_matrix.cols; ++i) {
                // because the branches are added by sampling, this array should be mostly sorted on its own
                // idk if std::sort takes full advantage of this, and that only a tail of the array is unsorted;
                // TODO
                auto &chance_data = matrix_node->chance_node_matrix.data[i];
                chance_data.branch_index = 0;
                std::sort(
                    chance_data.sorted_branches.begin(),
                    chance_data.sorted_branches.end(),
                    [](const Branch *x, const Branch *y) { return x->p > y->p; });
            }
        }

        void alpha_beta(MatrixNode *matrix_node, BaseData &base_data, HeadData &head_data, TempData &temp_data) const {

            head_data.step_forward();

            temp_data.state.get_actions();
            temp_data.rows = temp_data.state.row_actions.size();
            temp_data.cols = temp_data.state.col_actions.size();

            // only allocate for all next function calls, modify in other functions when necessary
            TempData next_temp_data;

            this->initialize_submatrix(matrix_node, base_data, head_data, temp_data, next_temp_data);

            while (temp_data.alpha < temp_data.beta && !temp_data.must_break) {

                auto [next_beta, next_alpha] = this->tentatively_solve_subgame(matrix_node, base_data, head_data, temp_data, next_temp_data);

                this->row_modify_beta_and_add_action(next_beta, matrix_node, base_data, head_data, temp_data, next_temp_data);
                this->col_modify_beta_and_add_action(next_alpha, matrix_node, base_data, head_data, temp_data, next_temp_data);

                if (temp_data.new_row_action && temp_data.new_col_action) {
                    this->solve_chance_node(matrix_node, base_data, head_data, temp_data, next_temp_data, 0, 0);
                }

                temp_data.alpha = std::min(next_alpha, temp_data.alpha);
                temp_data.beta = std::min(next_beta, temp_data.beta);

                temp_data.reset_flags_for_alpha_beta_loop();
            }

            this->sort_branches_and_reset(matrix_node);

            head_data.step_back();
        }

        struct Output {
            mpq_class alpha, beta;
            std::vector<size_t> counts{};
            std::vector<size_t> times{};
        };

        Output run(uint32_t depth, PRNG &device, const State &state, Model &model, MatrixNode &node) const {
            Output output;
            for (uint32_t d = 1; d <= depth; ++d) {
                BaseData base_data{depth, &device, &model, state, 0, 0, {}, {}};
                HeadData head_data{0, 0, 0, 0};
                TempData temp_data;
                const auto start = std::chrono::high_resolution_clock::now();
                this->alpha_beta(&node, base_data, head_data, temp_data);
                const auto end = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                output.alpha = temp_data.alpha;
                output.beta = temp_data.beta;
                // output.counts.push_back(node.count_matrix_nodes());
                output.counts.push_back(0);
                output.times.push_back(duration.count());
            }
            return output;
        }

        template <typename T>
        Output run(const std::vector<T> depths, PRNG &device, const State &state, Model &model, MatrixNode &node) const {
            Output output;
            for (const auto depth : depths) {
                BaseData base_data{depth, &device, &model, state, 0, 0, {}, {}};
                HeadData head_data{0, 0, 0, 0};
                TempData temp_data;
                const auto start = std::chrono::high_resolution_clock::now();
                this->alpha_beta(&node, base_data, head_data, temp_data);
                const auto end = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                output.alpha = temp_data.alpha;
                output.beta = temp_data.beta;
                // output.counts.push_back(node.count_matrix_nodes());
                output.counts.push_back(0);
                output.times.push_back(duration.count());
            }
            return output;
        }
    };
};
