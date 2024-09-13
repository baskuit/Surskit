
# State
### Markov Decision Process
A state is a two player Markov game when we are referring to a type. An instance of that type is also called a state and it corresponds to a 'state' of the Markov game. Usually we mean an instance when using this term.
A state is transitioned when joint actions are committed by the players. This is handled by the member function `apply_actions(Action, Action)`, which mutates the state into its successor. The 'Markov' descriptor means that these transitions are stochastic, but this behavior requires calling the member function `randomize_transition(PRNG &)`. Unless this function is called before `apply_actions`, a given state will always transition in the same way.
The MDP and perfect info conditions need the following types:
* `Action`
* `Obs`
* `Prob`

An object of `Obs` type is often called an observation and it is specified by the Markov condition.
The `Prob` type is the associated probability of that transition and it is not even required to be numeric.

### Chance Player
In the interface and docs we often formulate the stochasticity using a third 'chance player' that selects an action after the players. The `Obs` object given after joint actions are committed can then be thought of as a 'chance action', and the fixed distribution of all transitions from an after-state is then the 'strategy' of the chance player.

### Perfect Information and Inheritance
The assumption of perfect information means that the legal actions for both players at any state is public knowledge. We also require that the chance action is observed by both players after the transition. The full set of possible transitions after a given pair of joint actions is not known. However, players can always distinguish transition from one another by comparing the observation.
We do not require that all possible transitions are known because there may be many possibilities and calculating them non-trivial. Also the probability of an observation doesn't need to be known. The state will provide an object of type `Prob` that represents this, but this type could be something like `bool` instead 
The `IsState<>` concept formalizes the general former notion of a state, while `IsPerfectInfoState<>` is the one actually required by the majority of library code.
There is template class called `PerfectInfoState<TypeList>` that has data members pertaining to the related concept. We assume that all state types derive from this class, since then they will automatically inherit members like `obs`, `prob`, and methods like `get_payoff()`, `is_terminal()`.

### Row and Column

### Constant Sum
Currently Pinyon only partially supports games that are not constant-sum.
General sum games are supported in the sense that the `Value` type-specifier, which codifies the payoffs for both players, has a general sum and constant sum implementation. The type-specifier is expected to have a `IS_CONSTANT_SUM` boolean property in this regard.
They are not supported in the sense that general sum games do not have many of the properties that library functions take for granted.
While they do admit Nash equilibria, the payoffs for either player given by these equilibria are not unique. This means that such games do not have a well-defined 'value' like constant sum games do.

 


# Concepts/Interface
### IsStateTypes
```cpp
{
    const_state.is_terminal()
} -> std::same_as<bool>;
```
Self-explanatory.
```cpp
{
    const_state.get_payoff()
} -> std::same_as<typename Types::Value>;
```
The result is only valid when the state is terminal. Otherwise, it is likely to be uninitialized.
```cpp
{
    state.apply_actions(action, action)
} -> std::same_as<void>;
```
Without the condition of perfect-info we do not assume that an identifying observation is returned or that the probability of the transition is known. Hence the function returns `void`, and perfect-info states will provide this information with members.
```cpp
{
    const_state.get_actions(actions, actions)
} -> std::same_as<void>;
```
It is assumed that calculating actions is expensive so it must be done manually with this function.
```cpp
{
    state.randomize_transition(device)
} -> std::same_as<void>;
```
This function must be called before `apply_actions` otherwise the transition will be deterministic.
```cpp
{
    const_state.get_obs()
} -> std::same_as<const typename Types::Obs &>;
```
Returns constant reference to the observation from the last transition. May be a run-time error to call this on a state that has never been transitioned. This comes from `pkmn/engine`, where 


### IsPerfectInfoStateTypes
```cpp
{
    state.obs
} -> std::same_as<typename Types::Obs &>;
{
    state.prob
} -> std::same_as<typename Types::Prob &>;
```
These members are updated after calling `apply_actions`. The assumption of perfect info means that an observation is available to distinguish transitions. There is no consensus whether this also means that the probability of the transitions (a.k.a. the strategy of the chance player) is also known.
The requirement for states to produce a `Prob` value **will not restrict** games or environments that don't genuinely provide this information. The `Prob` is allowed to be a non-numeric type like `bool` in this case, and this choice does not affect the Tree Bandit methods since those are sample based. Therefore they do not use this data.
This possibility is not the only reason that `Prob` is a distinct type specifier from `Real`. If the probability is known, it is likely that it is a *rational* number rather than a floating point. This presents the opportunity to use the pseudo expression `total_prob == Types::Prob{1}` as a condition to test that all transitions have been observed.
```cpp
{
    state.terminal
} -> std::same_as<bool &>;
{
    state.payoff
} -> std::same_as<typename Types::Value &>;
```
The terminality of the state is stored as a bool and the method `is_terminal()` is implemented as a getter; Ditto for `get_payoff()`.
```cpp
{
    state.get_actions()
} -> std::same_as<void>;
{
    state.row_actions
} -> std::same_as<typename Types::VectorAction &>;
{
    state.col_actions
} -> std::same_as<typename Types::VectorAction &>;
```
These are essentially the same methods and attributes from the `IsState<>` concept, except now the data that was passed as return-arguments to those methods is now stored as members. 
This is largely a convenience, to spare the user from instantiating action vectors and values all the time. There should be very little overhead to this attachment. 

## IsChanceStateTypes
```cpp
{
    state.get_chance_actions(action, action, chance_actions)
} -> std::same_as<void>;
{
    state.apply_actions(action, action, obs)
} -> std::same_as<void>;
```
A 'chance state' is one where calculating the chance actions for a given pair of player actions is possible. It is not needed for tree bandit search since that is sample based, but it is required for solver algorithms. We assume that the probability that is observed is correct and not a `bool`-like placeholder. 
There is an overloaded `apply_actions` method where the user can specify the exact transition to occur by providing the same observation that the transition will produce as an argument (i.e. a 'chance action'.)
It is clear how these methods are required for solving since we need to be able to traverse the abstract game tree at will.

## IsSolvedStateTypes
```cpp
{
    state.get_strategies(strategy, strategy)
} -> std::same_as<void>;
```
A solved state is one where at least one pair of Nash equilibrium strategies is known and so is the associated payoff.
The strategies are provided by method above, and the Nash payoff is just the normal `payoff` member. As a consequence the `payoff` is now updated after every transition to reflect its current value at the new state.
This concept assumes that the game is constant sum.

## Subsumption
Each of these concepts assumes that the concepts before it are also satisfied. It is theoretically not necessary for a 'solved state' to also be a 'chance state' but it is almost guaranteed in practice. 

# Implementations
### MoldState
This is basically the simplest possible state in implementation. It's main purpose is testing and benchmarking as a control. It's only member is the `depth` parameter, which determines how many transitions are made until the state is terminal. It always has the same number of actions for both players, and so the `row_actions`, `col_actions` members are initialized in the `MoldState` constructor and never changed. Thus `get_actions` is a no-op and `apply_actions` merely decrements `depth` and checks if `depth == 0`. The payoff member is not changed or even initialized.

### OneSumMatrixGame
A basic one shot matrix game where the payoff sum is 1. It is non-stochastic, meaning there is only one possible transition for a pair of joint actions. It can be constructed by directly passing such a matrix, or passing a PRNG device and the number of rows and columns; the entries are then generated using (a copy of) the device.


### RandomTree

This class is a powerful and expressive way to create a random games for testing.
The game is well-defined, in the sense that providing the same hyper-parameters, helper functions, and PRNG device to the constructor will produce the exact same state. One advantage of this property is that arbitrarily large games can be used to test models via the `ModelBandit` utility. Previously, random trees had to be solved and the node tree was used as the state.

```cpp
State
(
    const Types::PRNG &device,
    size_t depth_bound,
    size_t rows,
    size_t cols,
    size_t transitions,
    Types::Q chance_threshold,
    int (*depth_bound_func)(State *, int),
    int (*actions_func)(State *, int),
    int (*payoff_bias_func)(State *, int)
)
```
Let's look at the full list of all constructor arguments:
* `device`
This device is copied and stored as a member. The member is used TODO
* `depth_bound`
This `int` member is an upper bound on the number of transitions before the state is terminal. The default behavior is to simply decrement this by one. This produces trees with uniform length.
* `rows`/`cols`
These member determine the number of actions each player has at a given state. It is updated during transition. The action type is simply `int` and the actions are always `0, 1, ..., rows - 1` for the row player etc.
* `transitions`
The maximum number of legal transitions at any given state. This number is const.
* `chance_threshold`
Upon initialization and after each transition, a new strategy for the chance player is computed. If the probability of any transition is less than this threshold, it is zeroed and the resulting distribution is re-normalized.

Function pointers:
Each of these functions determines how their namesake members change after each transition
* `depth_bound_func`
Decrements by one
* `actions_func`
No change
* `payoff_bias_func`
Increment by a randomly selected value from $\{-1, 0 ,1\}$.

The `grow-lib` header contains some alternatives to these functions that change the nature of the tree games. This collection is a WIP, and the only implemented will make the resulting games *alternative-move*. One player will only have a single actions is essentially a 'pass', and the passing player switches every transition.


### SolvedState

Any game which satisfies the `IsChanceStateTypes` concept can have it entire game tree solved using the `FullTraversal` algorithm. As the name implies, this will traverse the entire game tree and thus produces a sub-game perfect solution.
The `FullTraversal` algorithm will represent the game tree on heap using nodes.

### ModelBandit

This state uses the type erasure wrappers `W::Types`
to wrap each of the models uniformly and treat them each like an action.

The model bandit state only transitions once before it is terminal, which means it is essentially a bandit problem. The actions in this problem are W::Types::Models, and the payoff that is recieved is determined by the payoff of the underlying state.

The models' inference is used to perform a 'heavy rollout' on a copy of the *underlying* state.

When a `ModelBandit` state is randomized, the new seed is what will be provided to the initial state generating function.


# Wrapping a State

