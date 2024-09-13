
# TODO
remove all strong typing non-sense. god, that was a mistake.

# Types Struct

As alluded earlier, a type list object is just a struct with some type declarations for a certain set of names (`Real`, `Mutex`, `VectorReal`) etc. The `DefaultTypes` struct in the next section essentially lists the minimal collection of declaration that are used in the library. There are also some additional constraints on the individual types that defines an expected interface.
The use of concepts will mean this interface is auto-suggested when dealing with constrained template parameters.

# `DefaultTypes`

Below is the entire definition of the `DefaultTypes` type list.
The template parameters are demarcated from the actual type aliases with an underscore prefix.
The template parameters with the `template <typename...>` prefix are not simply types but are themselves templates.
```cpp
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
    using Q = _Rational;
    using Real = _Real;

    using Action = _Action;
    using Obs = _Obs;
    using ObsHash = ObsHashType<_Obs>;
    using Prob = _Prob;

    using Value = _Value<Real>;
    using VectorReal = _Vector<Real>;
    using VectorAction = _Vector<Action>;
    using VectorInt = _Vector<int>;
    using MatrixReal = _Matrix<Real>;
    using MatrixInt = _Matrix<int>;
    using MatrixValue = _Matrix<Value>;

    template <typename... Args>
    using Vector = _Vector<Args...>;
    template <typename... Args>
    using Matrix = _Matrix<Args...>;

    using Mutex = std::mutex;
    using PRNG = _PRNG;
    using Seed = _Seed;
};
```

## Wrapped Primitives

The `Real`, `Prob`, `Action`, and `Obs` type aliases that are defined in the `DefaultTypes` struct are not the same as the types which are provided in the template parameter list. Since these types are usually C++ primitives, they are instead wrapped with template classes which store the same data but also provide some extra functionality and make the search code more consistent when using library types like `mpq_class`. 

### `Wrapper<T>`

All the types in this group are derived from `Wrapper<T>`. 
* This purpose of this class is to simply hold a value of type `T`, and so any class which is derived from `Wrapper<T>` will automatically store this data by inheriting that member.
* If we could derive `Wrapper<T>` from `T` then we would not need to create a `T value` member since the derived class would inherit the `T` data. However, we cannot derive from primitive types in C++ so using a member is necessary.
*  This class defines a conversion to type `T`, which is basically an 'unwrapping' operation. This operator is marked as explicit, otherwise an implicit unwrapping might occur without the user's knowledge, which would circumvent the strong-typing conventions. The conversion operator is always invoked via a static cast, e.g. `double raw_data = static_cast<double>(some_real);`
* The underlying type `T` is accessible via an alias declaration `using type = T;`. This allows the user to make assertions about the underlying type easily, e.g. 
	```cpp
	if constexpr (std::is_same_v<typename Types::Real::type, mpq_class>) {
		mpq_canonicalize(some_real._value);
	}
	```
* The user should never construct or interface with this class directly. Its constructor is called by the constructors of its derived classes automatically. 

### `ArithmeticType<T>`

**This section is important to understand otherwise the user may find that expressions which look valid may cause a compilation error.**
```cpp
typename Types::Real x = 0; // valid
// ...
x += 1; // invalid, no match for operator += with...
```
The `Types::Real` and `Types::Prob` wrappers need to have the same arithmetic functionality as their underlying types. However, these operations would only be available a priori if the wrappers were *derived* from their underlying types. The class which serves as the default implementation of `Types::Vector` has this luxury, but C++ does not allow for a class to be derived from primitive types like `double`.

Thus we are forced to define each of these operations manually. We could define the common operations (`+`, `==`, etc) for `Real` and `Prob` separately, and we would then also have to define 'mixed' operations too (where we multiply a `Real` typed player payoff by a `Prob`).  However, the reduction of boiler-plate code is a core design principle.

Thus we define the operations on a new class `ArithmeticType<T>` and make `RealType<T>`, `ProbType<T>` derived from this class, so that they inherit these operations. This approach introduces its own kinks which have to be smoothed over.

```cpp
typename Types::Real x{1}, y{1};
x + y; // ArithmeticType<T>;
```
Both the operands are staticly cast to `ArithmeticType<T>`, which is also the return type of the operation. Thus the result has to be cast into either a `Real<T>` or a `Prob<T>`. We make this constructor explicit, again so that implicit conversions do not foil the strong typing convention.
```cpp
typename Types::Prob w {x * y + 1};
typename Types::Real z = x + y;
// note: the second line makes use of a specially defined assignment operator 
// RealType<T>::RealType<T> operator=(ArithmeticType<T>)
```

* Like `Wrapper<T>`, the user never needs to explicitly construct `ArithmeticType<T>`.

* Currently only right-handed operators are defined. That is, the expression `1 + x;` has no match but `x + 1` does. Thus we express `1 - x` (e.g. getting the column players payoff from the row players in a 1-sum game) as `x * -1 + 1`. This is perhaps an unreasonable aversion to repetitive code.

* Applying a non-elementary operation like `std::exp` is done by a static cast conversion to primitive, e.g. 
`const  typename  Types::Real  y{std::exp(static_cast<double>(gains[i] *  eta))};` 

### `RealType<T>` & `ProbType<T>`

Since the essential mathematical operators are defined on their base class, there is currently only one task that these two are relied upon to perform.
The default implementation for multiple precision arithmetic is provided by `mpq_class`, which is the C++ front-end for GMP, or the [GNU Multiple Precision Library](https://gmplib.org/).
Unfortunately, there is one quirk of this class that makes writing polymorphic code (or more specifically code that works the same for float and `mpq_class` without template specialization or `if constexpr` everywhere) difficult.
In order for the equality and comparison (`>=` etc) operators to work correctly, the `mpq_class` objects must be *canonicalized*, or reduced to a proper fraction. Unless the user naively initializes an `mpq_class` to be un-canonical (e.g. `typename Types::Q {2, 4}`), the only way rationals end up improper is after an arithmetic operation. The GMP library is designed for performance and thus does not perform this reduction automatically.
Our method of handling this is to automatically canonicalize the underlying value when an `ArithmeticType<mpq_class>` is cast to a `RealType<mpq_class>` or `ProbType<mpq_class>`, since that typically means we are finished operating on that value.
```cpp
RealType &operator=(const ArithmeticType<T> &val) {
    this->_value = val._value;
    if constexpr (std::is_same_v<T, mpq_class>) {
        mpq_canonicalize(this->value.get_mpq_t());
    }
    return *this;
}
```
Regrettably, this means that some unnecessary reductions may be performed. However, I feel that `mpq_class` is restricted to theoretical or academic contexts, rather than in the performance critical context of a user created engine, where they will instead surely use `double` or `float` as the underlying `Real` and `Prob` types.


### `ObsType<T>`
A wrapper for the Obs type. Only the equality operator `==` is used on this, to check whether a particular state transition has occurred before. This is how the search functions match the development of a state accurately in the corresponding tree structure.

### `ActionType<T>`
A wrapper for the action type, and the argument type of the `apply_actions` method. The methods at for this wrapper are for convenience. The assignment operator means that `row_action = 0` can be used instead of `row_action = typename Types::Action{0}`.


## Vectors

In most cases, `std::vector` is an adequate container for storing floats, actions, etc.

However, vectors are heavier than `std::array`, which does not reallocate or allocate unneeded storage and which size is known at compile-time. This means standard arrays are preferable in certain cases, for example if a game always has the same number of actions.

Thus we use `VectorReal`, `VectorInt`, `VectorAction` to specify the most common uses for contiguous containers.

We rely on the methods `resize` and `clear` for the most part. Resize is used to fully initialize a vector, which does pose some risk. There are no checks that the vector being passed is empty, and these pre-existing members won't be overwritten with the intended value (second argument to `resize`.)

### Arrays
We define an array-based container class with the same vector-like interface that Pinyon relies on.
```cpp
template <size_t MaxSize>
struct A
{
    template <typename T>
    struct Array : std::array<T, MaxSize>
    {
        // ...
```

The `A` outer class is a trick to give the `A::Array` template a parameter signature that is compatible with `std::vector`. The `size` parameter is attached to the template  `A`, not `Array`.
The de facto size of the container is stored as the member `Array<..>::_size` and connected the redefined `begin()`, `end()`, `resize()` methods in the obvious way. This allows us to use range-based iteration (e.g. `for (auto x : array)`) properly.  

## Matrices

Obviously a library for matrix tree games should have a matrix type. In practice though its mostly a data class, used to store visit and reward information for the MatrixUCB algorithm and its variants. There is currently no need for matrix multiplication. The most relevant use-case is the calculation of exploitability, since the expected rewards can be formulated using matrix multiplication. However, in practice I've found it's faster to use another approach.

Much like Vectors, we have specially denominated types for common data types: `MatrixReal`, `MatrixInt`, `MatrixValue`

```cpp
{
    matrix.clear()
} -> std::same_as<void>;
{
    matrix.operator[](0)
} -> std::same_as<T &>;
```
The only implementation of `Matrix<T>` is derived from `std::vector<T>`. The values for the entries are stored in row-major order. Library functions frequently use `std::vector<T>::operator[]()` to iterate over entries.
```cpp
{
    Matrix{0, 0}
} -> std::same_as<Matrix>;
{
    matrix.fill(0, 0)
} -> std::same_as<void>;
```
Since matrices are usually reserved for search stats, they are initialized after construction. The `fill` method is the 2D equivalent to `resize` and is used in initialization much like `resize` is for vectors. In the case that we can properly initialize on construction, the constructor takes the number of rows and columns as arguments.
```cpp
{
    matrix.get(0, 0)
} -> std::same_as<T &>;
{
    const_matrix.get(0, 0)
} -> std::same_as<const T &>;
```
Standard matrix accessor.

## PRNG and Seed
Abbreviation of pseudo-random number generator. Instances are usually called "device".
All (single-threaded) operations of Pinyon are intended to be deterministic.
```cpp
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
};
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
```
Most of these constrains are self-explanatory. 
`PRNG` is copy constructable. It also copies the state of the device. Copying a device from the beginning is done via `PRNG{old_device.get_seed()}`.
The `discard(n)` operation advances the state of the device `n` times. It is used in the random tree class.
A seed is the canonical way to construct a `PRNG`.

## Mutex
```cpp
{
    mutex.lock()
} -> std::same_as<void>;
{
    mutex.unlock()
} -> std::same_as<void>;
{
    mutex.try_lock()
} -> std::same_as<bool>;
```
The standard library mutex is unnecessarily large on most systems and so the library includes a simple spin lock implementation. The required interface is minimal and self-explanatory.

## Value

Pinyon does not make the assumption that games are constant sum. Although all games of interest have this property, it is not always satisfied by every ancillary game.

In each matrix node, MatrixUCB must store a cumulative score for each of the `rows * cols` joint actions. Without the constant sum assumption, this requirement doubles.

Thus we use a special struct to payoffs/value estimates for a game. This struct stores a `Real` type member for the row player, and if the game is not constant sum then stores an additional `Real` for the column player.

```cpp
Real Value::get_row_value() const;
Real Value::get_col_value() const;
static Real Value::PAYOFF_SUM;  
```


> Warning: Although the constant sum version of the struct is smaller, it usually result in slightly less performance than its more general counterpart. The reason for this lies in the backward phase of tree bandit, where we update the matrix and chance nodes using `get_col_value()`. In the non constant sum version, the column players value estimate at the leaf node is simply retrieved. In the constant sum version, it must be recalculated at each visited node via `PAYOFF_SUM - get_row_value()`.

## Q

```cpp
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
};
```
Simple rational type that can initialize `Real`, `Prob` regardless of their underlying type.
Not guaranteed to be arbitrary precision.

## Strategy

TODO
This type is currently unused. For bandit algorithms that use the model's policy inference (MatrixPUCB), the policy for both players is stored in the matrix node stats. Using even a vector of doubles here would explode the size of the matrix nodes, resulting in a significant performance drop. 

Since Prob distributions satisfy `0 < p_i < 1` for all entries `p_i` and precision is not terribly important, we can quantize the distribution using unsigned integers.

For example, `uint8_t` has 256 possible values. Thus  the distro `{0.3, 0.3, 0.4}` could be represented as
```cpp
uint8_t strategy[3] = {76, 76, 104};
```

This is a 4x reduction in size from `float` already and we could even omit the last entry, since the probabilities must sum to 1.
