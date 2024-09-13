#pragma once

#include <libpinyon/math.h>
#include <types/random.h>
#include <types/array.h>
#include <types/matrix.h>
#include <types/random.h>
#include <types/value.h>
#include <types/mutex.h>
#include <any>

/*

TypeList

*/

template <typename T>
struct ObsHashType
{
    size_t operator()(const T &obs) const
    {
        return static_cast<T>(obs); // TODO
    }
};

template <
    typename _Real,
    typename _Action,
    typename _Obs,
    typename _Prob,

    template <typename...> typename _Value = PairReal,
    template <typename...> typename _Vector = std::vector,
    template <typename...> typename _Matrix = Matrix,

    typename _Mutex = std::mutex,
    typename _Seed = uint64_t,
    typename _PRNG = prng,
    typename _Rational = Rational<int>>
struct DefaultTypes
{
    using TypeList = DefaultTypes;
    // I don't know how to deduce a 'minimal typelist' in a generic way
    // for things like Search and Model, so we need to provide some declarations
    // There are not too many types so we avoid boilerplate `using ModelTypes = ` etc
    // by providing just `Types::TypeList`

    using Q = _Rational;
    using Real = _Real;

    using Action = _Action;
    using Obs = _Obs;
    using ObsHash = ObsHashType<_Obs>;
    using Prob = _Prob;
    // These are the wrappers for strong typing

    using Value = _Value<Real>;
    using VectorReal = _Vector<Real>;
    using VectorAction = _Vector<Action>;
    using VectorInt = _Vector<int>;
    using MatrixReal = _Matrix<Real>;
    using MatrixInt = _Matrix<int>;
    using MatrixValue = _Matrix<Value>;
    // Portmanteau names to avoid
    // `Types::template Matrix<typename Types::Real>`

    template <typename... Args>
    using Vector = _Vector<Args...>;
    template <typename... Args>
    using Matrix = _Matrix<Args...>;

    using Mutex = std::mutex;
    using PRNG = _PRNG;
    using Seed = _Seed;
};

template <typename T>
concept IsArithmetic = requires(T x) {
    static_cast<T>(x + x);
    static_cast<T>(x - x);
    static_cast<T>(x * x);
    static_cast<T>(x / x);
};

template <typename Obs>
concept IsObs = requires(Obs obs) {
    {
        obs == obs
    } -> std::same_as<bool>;
    // Obs type is just a small and sure way to identify distinct transitions
    // of a State after commiting the same joint actions
};
template <typename ObsHash, typename Obs>
concept IsObsHash = requires(Obs obs) {
    true;
}; // TODO

template <typename Value, typename Real>
concept IsValue = requires(Value &value) {
    {
        value.get_row_value()
    } -> std::same_as<Real>;
    {
        value.get_col_value()
    } -> std::same_as<Real>;
    {
        value = value
        // copy assignable
    } -> std::same_as<Value &>;
};

template <typename Mutex>
concept IsMutex = requires(Mutex &mutex) {
    {
        mutex.lock()
    } -> std::same_as<void>;
    {
        mutex.unlock()
    } -> std::same_as<void>;
    {
        mutex.try_lock()
    } -> std::same_as<bool>;
};

template <typename Vector, typename T>
concept IsVector = requires(Vector &vector, T &value) {
    {
        vector.operator[](0)
    } -> std::same_as<T &>;
    {
        vector.resize(0)
    } -> std::same_as<void>;
    {
        vector.resize(0, value)
    } -> std::same_as<void>;
    {
        vector.clear()
    } -> std::same_as<void>;
} && std::ranges::sized_range<Vector>;

template <typename Matrix, typename T>
concept IsMatrix = requires(Matrix &matrix, const Matrix &const_matrix, T &value) {
    {
        matrix.clear()
    } -> std::same_as<void>;
    {
        matrix.operator[](0)
    } -> std::same_as<T &>;
    {
        Matrix{0, 0}
    } -> std::same_as<Matrix>;
    {
        matrix.fill(0, 0)
    } -> std::same_as<void>;
    {
        matrix.get(0, 0)
    } -> std::same_as<T &>;
    {
        const_matrix.get(0, 0)
    } -> std::same_as<const T &>;
};

template <typename PRNG, typename Seed>
concept IsPRNG = requires(PRNG &device, const PRNG &const_device, Seed seed) {
    {
        device = const_device
    } -> std::same_as<PRNG &>;
    {
        const_device.get_seed()
    } -> std::same_as<Seed>;
    {
        device.random_seed()
    } -> std::same_as<Seed>;
    {
        PRNG{seed}
    } -> std::same_as<PRNG>;
    {
        device.random_int(0)
    } -> std::convertible_to<int>;
    {
        device.uniform()
    } -> std::same_as<double>;
    {
        device.discard(0)
    } -> std::same_as<void>;
    {
        device.sample_pdf(std::vector<std::any>{})
    } -> std::convertible_to<int>;
};

template <typename Q>
concept IsRational = requires(Q &x) {
    {
        Q{1, 1}
    };
    {
        x.canonicalize()
    } -> std::same_as<void>;
    {
        std::convertible_to<Q, float>
    };
    {
        std::convertible_to<Q, mpq_class>
    }; // these last two are the whole point of the rational type.
};

template <typename Types>
concept IsTypeList =
    IsObs<typename Types::Obs> &&
    IsArithmetic<typename Types::Real> &&
    IsArithmetic<typename Types::Prob> &&
    IsValue<typename Types::Value, typename Types::Real> &&
    IsPRNG<typename Types::PRNG, typename Types::Seed> &&
    IsMutex<typename Types::Mutex> &&
    IsVector<typename Types::VectorReal, typename Types::Real> &&
    IsVector<typename Types::VectorInt, int> &&
    IsVector<typename Types::VectorAction, typename Types::Action> &&
    IsMatrix<typename Types::MatrixReal, typename Types::Real> &&
    IsMatrix<typename Types::MatrixValue, typename Types::Value> &&
    IsMatrix<typename Types::MatrixInt, int>;

/*

Instances

*/

using SimpleTypes = DefaultTypes<
    double,
    int,
    int,
    double>;

using RandomTreeFloatTypes = DefaultTypes<
    double,
    int,
    int,
    double,
    ConstantSum<1, 1>::Value>;

using RandomTreeRationalTypes = DefaultTypes<
    mpq_class,
    int,
    int,
    mpq_class,
    ConstantSum<1, 1>::Value>;

using SimpleTypesSpinLock = DefaultTypes<
    double,
    int,
    int,
    double,
    PairReal,
    std::vector,
    Matrix,
    spinlock>;
