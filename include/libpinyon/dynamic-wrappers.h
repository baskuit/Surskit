#pragma once

#include <algorithm/algorithm.h>
#include <tree/tree.h>

#include <memory>

/*

Type erasure using unique_ptr. For any tree-bandit-like type list, its State, Model, Search, and MatrixNode types
can be wrapped as their W::Types counterparts.

These types have `get_actions`, `inference` etc as virtual methods. If there is an underlying type mismatch
there are no checks to prevent the nullptr deference.

*/

namespace TypeListNormalizer
{

    /*

    Even if two State types are equal, e.g. `Types1 = MonteCarloModel<MoldState<2>>::State` and `Types2 = MoldState<2>::State`,
    the StateT<>, ModelT<>, SearchT<> classes might not be convertible. That is,

    StateT<MoldState<2>>, StateT<MonteCarloModel<MoldState<2>>> might not be compatible.
    This would basically defeat the purpose of wrapping, so it has to be addressed.

    The solution is the 'normalize' the typelists so that the resulting StateT<> instantiations are equal as long as their primary
    and ancillary data types are equal.

    W::Types is a specific type list that is should allow every other type list to be 'converted' to it via wrapping
    this is assumed a few things about type lists
    e.g. Reals are assumed to have a conversion to double, get_actions() output has fixed ordering (see below), etc
    mpq_class can't convert, it uses get_d(), so we handle this at the level of the wrapper by adding conversion operator
    this unfortunatley means the library won't immediately compile if the RealType etc. wrappers are removed
    although its very close

    The only StateT etc. constructor is templated and an instance of that *type list* is used as the first parameter.
    The make_state etc. factory functions are the final layer so that this type list can be provided via a template parameter list
    e.g. auto state = make_state<TreeBandit<...>>(// ...args)
    and without constructing the type list, which is verbotten


    _ in template types just to not shadow the using declaration
    _ prefix used for virtual methods in Dynamic namespace,
    because they mirror but are not related to their name-likes in the static realm

    */

    // State
    template <
        typename _Real,
        typename _PRNG,
        typename _State>
    struct MinimalStateTypes
    {
        using Real = _Real;
        using PRNG = _PRNG;
        using State = _State;
    };

    template <typename Types>
    using MStateTypes = MinimalStateTypes<
        typename Types::Real,
        typename Types::PRNG,
        typename Types::State>;

    // Model
    template <
        typename _Real,
        typename _PRNG,
        typename _State,
        typename _Model,
        typename _ModelOutput>
    struct MinimalModelTypes
    {
        using Real = _Real;
        using PRNG = _PRNG;
        using State = _State;
        using Model = _Model;
        using ModelOutput = _ModelOutput;
    };

    template <typename Types>
    using MModelTypes = MinimalModelTypes<
        typename Types::Real,
        typename Types::PRNG,
        typename Types::State,
        typename Types::Model,
        typename Types::ModelOutput>;

    // Search
    template <
        typename _Real,
        typename _PRNG,
        typename _VectorReal,
        typename _State,
        typename _Model,
        typename _ModelOutput,
        typename _MatrixNode,
        typename _Search>
    struct MinimalSearchTypes
    {
        using Real = _Real;
        using PRNG = _PRNG;
        using VectorReal = _VectorReal;
        using State = _State;
        using Model = _Model;
        using ModelOutput = _ModelOutput;
        using MatrixNode = _MatrixNode;
        using Search = _Search;
    };

    template <typename Types>
    using MSearchTypes = MinimalSearchTypes<
        typename Types::Real,
        typename Types::PRNG,
        typename Types::VectorReal,
        typename Types::State,
        typename Types::Model,
        typename Types::ModelOutput,
        typename Types::MatrixNode,
        typename Types::Search>;
};

namespace W
{
    struct State;
    struct Model;
    struct Search;
    struct MatrixNode;

    struct Types : SimpleTypes
    // as long as the order of output of get_actions is fixed we can equate actions with their indices
    {
        using ModelInput = W::State;
        struct ModelOutput
        {
            PairReal<double> value;
            std::vector<double> row_policy, col_policy;
        };
        using State = W::State;
        using Model = W::Model;
        using Search = W::Search;
        using MatrixNode = W::MatrixNode;
    };

    namespace Dynamic
    {
        template <typename T>
        struct StateT;
        struct State
        {
            virtual ~State() = default;
            virtual std::unique_ptr<Dynamic::State> clone() const = 0;
            virtual void _get_actions(size_t &, size_t &) = 0;
            virtual void _apply_actions(Types::Action, Types::Action) = 0;
            virtual bool _is_terminal() const = 0;
            virtual Types::Value _get_payoff() const = 0;
            virtual void _randomize_transition(Types::PRNG &) = 0;
        };
        template <typename T>
        struct StateT : State
        {
            typename T::State data;

            template <typename... Args>
            StateT(const Args &...args) : data{args...} {}

            std::unique_ptr<Dynamic::State> clone() const
            {
                return std::make_unique<StateT>(*this);
            }

            void _get_actions(size_t &rows, size_t &cols)
            {
                data.get_actions();
                rows = data.row_actions.size();
                cols = data.col_actions.size();
            };
            void _apply_actions(typename Types::Action row_idx, typename Types::Action col_idx)
            {
                data.apply_actions(
                    data.row_actions[static_cast<int>(row_idx)],
                    data.col_actions[static_cast<int>(col_idx)]);
            };
            bool _is_terminal() const
            {
                return data.is_terminal();
            };
            Types::Value _get_payoff() const
            {
                if constexpr (std::is_same_v<mpq_class, typename T::Real>)
                {
                    return Types::Value{
                        static_cast<double>(data.get_payoff().get_row_value().get_d()),
                        static_cast<double>(data.get_payoff().get_col_value().get_d())};
                }
                else
                {
                    return Types::Value{
                        static_cast<double>(data.get_payoff().get_row_value()),
                        static_cast<double>(data.get_payoff().get_col_value())};
                }
            }
            void _randomize_transition(Types::PRNG &device)
            {
                typename T::PRNG local_device{device.uniform_64()};
                return data.randomize_transition(local_device);
            }
        };

        struct Model
        {
            virtual ~Model() = default;
            virtual std::unique_ptr<Dynamic::Model> clone() const = 0;
            virtual void _inference(const Dynamic::State *, Types::ModelOutput &) = 0;
        };
        template <typename T>
        struct ModelT : Model
        {
            typename T::Model data;

            template <typename... Args>
            ModelT(const Args &...args) : data{args...} {}

            std::unique_ptr<Dynamic::Model> clone() const
            {
                return std::make_unique<ModelT>(*this);
            }

            void _inference(
                const Dynamic::State *state_ptr,
                Types::ModelOutput &output)
            {
                const typename T::State &state = dynamic_cast<const Dynamic::StateT<TypeListNormalizer::MStateTypes<T>> *>(state_ptr)->data;
                typename T::ModelOutput output_;
                data.inference(typename T::State{state}, output_); // rvalue is copied state TODO does this work???
                output.value = Types::Value{
                    static_cast<double>(output_.value.get_row_value()),
                    static_cast<double>(output_.value.get_col_value())};
                // TODO
                output.row_policy.resize(state.row_actions.size());
                output.col_policy.resize(state.col_actions.size());

                for (int i = 0; i < state.row_actions.size(); ++i)
                {
                    output.row_policy[i] = static_cast<double>(output_.row_policy[i]);
                }
                for (int i = 0; i < state.col_actions.size(); ++i)
                {
                    output.col_policy[i] = static_cast<double>(output_.col_policy[i]);
                }
            }
        };

        struct MatrixNode
        {
            virtual ~MatrixNode() = default;
            virtual std::unique_ptr<Dynamic::MatrixNode> create_new() const = 0;
        };
        template <typename T>
        struct MatrixNodeT : MatrixNode
        {
            typename T::MatrixNode data{};

            std::unique_ptr<Dynamic::MatrixNode> create_new() const
            {
                return std::make_unique<MatrixNodeT>();
            }
        };

        struct Search
        {
            virtual ~Search() = default;
            virtual std::unique_ptr<Dynamic::Search> clone() const = 0;
            virtual std::unique_ptr<Dynamic::MatrixNode> get_matrix_node() const = 0;
            virtual size_t _run(
                size_t duration_ms,
                Types::PRNG &device,
                Dynamic::State *state_ptr,
                Dynamic::Model *model_ptr,
                Dynamic::MatrixNode *matrix_node_ptr) = 0;
            virtual size_t _run_for_iterations(
                size_t duration_ms,
                Types::PRNG &device,
                Dynamic::State *state_ptr,
                Dynamic::Model *model_ptr,
                Dynamic::MatrixNode *matrix_node_ptr) = 0;
            virtual void _get_empirical_strategies(
                Dynamic::MatrixNode *matrix_node_ptr,
                Types::VectorReal &row_strategy,
                Types::VectorReal &col_strategy) const = 0;
        };
        template <typename T>
        struct SearchT : Search
        {
            typename T::Search data;

            template <typename... Args>
            SearchT(const Args &...args) : data{args...} {}

            std::unique_ptr<Dynamic::Search> clone() const
            {
                return std::make_unique<SearchT>(*this);
            }

            std::unique_ptr<Dynamic::MatrixNode> get_matrix_node() const
            {
                return std::make_unique<MatrixNodeT<TypeListNormalizer::MSearchTypes<T>>>();
            }

            size_t _run(
                size_t duration_ms,
                Types::PRNG &device,
                Dynamic::State *state_ptr,
                Dynamic::Model *model_ptr,
                Dynamic::MatrixNode *matrix_node_ptr) const
            {
                typename T::PRNG device_{device.uniform_64()};
                typename T::State &state = dynamic_cast<StateT<TypeListNormalizer::MStateTypes<T>> *>(state_ptr)->data;
                typename T::Model &model = dynamic_cast<ModelT<TypeListNormalizer::MModelTypes<T>> *>(model_ptr)->data;
                typename T::MatrixNode &matrix_node = dynamic_cast<MatrixNodeT<T> *>(matrix_node_ptr)->data;
                // T is a normalized typelist, and we use the same one for the matrix node as we do for the search
                const size_t iterations = data.run(duration_ms, device_, state, model, matrix_node);
                return iterations;
            }
            size_t _run_for_iterations(
                size_t iterations,
                Types::PRNG &device,
                Dynamic::State *state_ptr,
                Dynamic::Model *model_ptr,
                Dynamic::MatrixNode *matrix_node_ptr) const
            {
                typename T::PRNG device_{device.uniform_64()};
                typename T::State &state = dynamic_cast<StateT<TypeListNormalizer::MStateTypes<T>> *>(state_ptr)->data;
                typename T::Model &model = dynamic_cast<ModelT<TypeListNormalizer::MModelTypes<T>> *>(model_ptr)->data;
                typename T::MatrixNode &matrix_node = dynamic_cast<MatrixNodeT<TypeListNormalizer::MSearchTypes<T>> *>(matrix_node_ptr)->data;
                const size_t ms = data.run_for_iterations(iterations, device_, state, model, matrix_node);
                return ms;
            }
            void _get_empirical_strategies(
                Dynamic::MatrixNode *matrix_node_ptr,
                Types::VectorReal &row_strategy,
                Types::VectorReal &col_strategy) const
            {
                const typename T::MatrixNode &matrix_node = dynamic_cast<MatrixNodeT<T> *>(matrix_node_ptr)->data;
                typename T::VectorReal r, c;
                data.get_empirical_strategies(matrix_node.stats, r, c);

                row_strategy.resize(r.size()); // TODO check this as well
                col_strategy.resize(c.size());
                std::copy(r.begin(), r.end(), row_strategy.begin());
                std::copy(c.begin(), c.end(), col_strategy.begin());
            }
        };

    }

    struct State : PerfectInfoState<Types>
    {
        std::unique_ptr<Dynamic::State> ptr;

        template <typename T, typename... Args>
        explicit State(T, const Args &...args) : ptr{std::make_unique<Dynamic::StateT<TypeListNormalizer::MStateTypes<T>>>(args...)} {}

        State(const State &other)
        {
            ptr = other.ptr->clone();
        }

        State &operator=(const State &other)
        {
            ptr = other.ptr->clone();
            return *this;
        }

        template <typename T>
        auto unwrap() const
        {
            return dynamic_cast<Dynamic::StateT<TypeListNormalizer::MStateTypes<T>> *>(ptr.get())->data;
        }

        bool is_terminal() const
        {
            return ptr->_is_terminal();
        }

        Types::Value get_payoff() const
        {
            return ptr->_get_payoff();
        }

        void get_actions(
            Types::VectorAction &row_actions,
            Types::VectorAction &col_actions) const
        {
            size_t rows, cols;
            ptr->_get_actions(rows, cols);
            row_actions.resize(rows);
            col_actions.resize(cols);
            for (int i = 0; i < rows; ++i)
            {
                row_actions[i] = i;
            }
            for (int i = 0; i < cols; ++i)
            {
                col_actions[i] = i;
            }
        };

        void randomize_transition(
            Types::PRNG &device)
        {
            return ptr->_randomize_transition(device);
        }

        void get_actions()
        {
            size_t rows, cols;
            ptr->_get_actions(rows, cols);
            this->init_range_actions(rows, cols);
        }

        void apply_actions(
            Types::Action row_action,
            Types::Action col_action)
        {
            ptr->_apply_actions(row_action, col_action);
        }
    };

    struct Model
    {
        std::unique_ptr<Dynamic::Model> ptr;

        template <typename T, typename... Args>
        explicit Model(T, const Args &...args) : ptr(std::make_unique<Dynamic::ModelT<TypeListNormalizer::MModelTypes<T>>>(args...)) {}

        Model(const Model &other)
        {
            ptr = other.ptr->clone();
        }

        Model &operator=(const Model &other)
        {
            ptr = other.ptr->clone();
            return *this;
        }

        template <typename T>
        auto unwrap() const
        {
            return dynamic_cast<Dynamic::ModelT<TypeListNormalizer::MModelTypes<T>> *>(ptr.get())->data;
        }

        void inference(
            const Types::ModelInput &input,
            Types::ModelOutput &output)
        {
            Dynamic::State *state_ptr = input.ptr.get();
            ptr->_inference(state_ptr, output);
        }

        void get_input(
            const Types::State &state,
            Types::ModelInput &input)
        {
            input = state;
        }
    };

    struct MatrixNode
    {
        std::unique_ptr<Dynamic::MatrixNode> ptr;

        MatrixNode(std::unique_ptr<Dynamic::MatrixNode> ptr) : ptr{std::move(ptr)} {}

        template <typename T>
        MatrixNode(T) : ptr{std::make_unique<Dynamic::MatrixNodeT<TypeListNormalizer::MSearchTypes<T>>>()} {}

        template <typename T>
        auto unwrap() const
        {
            return dynamic_cast<Dynamic::MatrixNodeT<TypeListNormalizer::MSearchTypes<T>> *>(ptr.get())->data;
        }
    };

    struct Search
    {
        std::unique_ptr<Dynamic::Search> ptr;

        template <typename T, typename... Args>
        explicit Search(T, const Args &...args) : ptr(std::make_unique<Dynamic::SearchT<TypeListNormalizer::MSearchTypes<T>>>(args...)) {}

        Search(const Search &other)
        {
            ptr = other.ptr->clone();
        }

        Search &operator=(const Search &other)
        {
            ptr = other.ptr->clone();
            return *this;
        }

        template <typename T>
        auto unwrap() const
        {
            return dynamic_cast<Dynamic::SearchT<TypeListNormalizer::MSearchTypes<T>> *>(ptr.get())->data;
        }

        MatrixNode get_matrix_node() const
        {
            return MatrixNode{ptr->get_matrix_node()};
        }

        size_t run(
            size_t duration_ms,
            Types::PRNG &device,
            Types::State &state,
            Types::Model &model,
            Types::MatrixNode &matrix_node)
        {
            return ptr->_run(duration_ms, device, state.ptr.get(), model.ptr.get(), matrix_node.ptr.get());
        }

        size_t run_for_iterations(
            size_t iterations,
            Types::PRNG &device,
            Types::State &state,
            Types::Model &model,
            Types::MatrixNode &matrix_node)
        {
            return ptr->_run_for_iterations(iterations, device, state.ptr.get(), model.ptr.get(), matrix_node.ptr.get());
        }

        void get_empirical_strategies(
            Types::MatrixNode &matrix_node,
            Types::VectorReal &row_strategy,
            Types::VectorReal &col_strategy) const
        {
            ptr->_get_empirical_strategies(matrix_node.ptr.get(), row_strategy, col_strategy);
        }
    };

    /*

    These functions are the intended interface.

    */

    template <typename TypeList, typename... Args>
    Types::State make_state(const Args &...args)
    {
        return Types::State{TypeList{}, args...};
    }

    template <typename TypeList, typename... Args>
    Types::Model make_model(const Args &...args)
    {
        return Types::Model{TypeList{}, args...};
    }

    template <typename TypeList, typename... Args>
    Types::Search make_search(const Args &...args)
    {
        return Types::Search{TypeList{}, args...};
    }

    template <typename TypeList>
    Types::MatrixNode make_node()
    {
        return Types::MatrixNode{TypeList{}};
    }

}
