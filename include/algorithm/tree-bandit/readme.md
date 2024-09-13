
# Tree Bandit

### Inheritance Pattern

The algorithms in the folder are generalizations of MCTS, which we call 'tree bandit'.

> In mathematics, "Monte Carlo" refers to estimation via random process. MCTS gets its name sake from the random rollout method of value estimation.

 The tree bandit algorithms are defined in two parts. 

* First there is the bandit algorithm, which takes the model type is its sole template parameter. This component decides how the algorithm chooses the next node in the tree to visit. It defines its own tree statistics for this purpose, called `MatrixStats` and `ChanceStats`, as well as its rules for updating these statistics.

* Secondly, there is the tree algorithm. This component wraps the bandit algorithm by accepting it as its first template parameter. It also accepts template parameters for the types/implementation of the matrix and chance nodes, as well as a boolean that determines whether the algorithms stops exploring the tree when it reaches an unexpanded node. This alternative stop-condition allows the tree search code to be used for replay generation.

# BanditAlgorithm

Bandit algorithms get their name from 'multi-armed bandits'. To quote Wikipedia:

> The multi-armed bandit problem models an agent that simultaneously attempts to acquire new knowledge (called "exploration") and optimize their decisions based on existing knowledge (called "exploitation"). The agent attempts to balance these competing tasks in order to maximize their total value over the period of time considered.

## A Terse Theoretical Overview

There is an entire field of mathematics devoted to describing and analyzing algorithms for multi-armed bandits. This field is itself split depending on the assumptions that are made about the nature of the problem and its rewards. The two most studied of these sub-fields are *stochastic bandits* and *adversarial bandits*.
 
 The standard formula in Monte Carlo tree search

$$ \frac{w_i}{n_i} + c \sqrt{\frac{\ln N_i}{n_i}} $$

is often called UCB (Upper Confidence Bounds). It is a solution to the stochastic bandits problem. By this, we mean that the rewards that are received are more or less "fixed". More precisely, the reward that is received is sampled from some fixed (but unknown) distribution. Games like Chess and Go fall under this description, where to distribution represents the minimax value of the position we are searching. In the slot-machine scenario that gives 'bandits' its name, this distribution, or its mean rather, represents some underlying 'profitability' of that machine.

In contrast, the rewards for adversarial bandits do not have to behave so nicely. The term adversarial here is meant to conjure the image of some foe that is able to change the rewards of the actions however they like, so as to foil our attempts to maximize our total value. Pokemon and other simultaneous move games match this definition. A move can be good or bad, depending entirely on what move the opponent plays, which we have no knowledge or say over.

The Exp3 algorithm is the most famous algorithm for decision making in this scenario. In comparison to UCB, it plays in a much more explorative manner. This makes it's play harder to exploit by the opponent, but it is thus much less able to exploit a 'predictable' opponent.

### Nash Equilibrium

In alternating turn games, the solution concept is understood by most players intuitively. Positions allow for 'best moves', and this is made precise by the notion of 'minimax'.
In simultaneous move games, and imperfect information in general, the concept of perfect play is codified by the notion of Nash equilibrium. Instead of a best move, which we can play deterministically and be sure that we are optimizing our reward, we have to settle for a *mixed strategy*. This is a fixed probability distribution over our actions, and we play a move by randomly sampling from this distribution.
Adversarial bandits 
Years ago, I wrote a casual summary of perfect play in Pokemon, which can be found in the docs directory.

### Not all bandits are created equal

In many schemes, the empirical move selection of the bandit algorithm is normalized and interpreted as a probability distribution. This strategy is treated as the output of the search, and in alternating move games, the action with the highest probability in this strategy is thought of as the best move.
> It is well-known that the empirical strategy of a stochastic bandit algorithm **does not converge** to Nash equilibrium when applied to simultaneous move games. On the contrary, an adversarial bandit algorithm, which guarantees 'sub-linear regret' in this setting, **will converge** to a Nash equilibrium. 

### MatrixUCB

The Exp3 and MatrixUCB algorithms are already provided. Not all bandits algorithms (i.e. stochastic bandit algorithms) are sound choices. Refer to "Analysis of Hannan Consistent Selection for Monte Carlo Tree Search in Simultaneous Move Games".

## Structs
* `MatrixStats`
Contains the stats the algorithm needs to use for the selection and update process. This minimal example only needs to store the number of actions for the players, but something like `exp3` would need to store the exponential weights and probably also some other hyper-parameters.
This struct should be as small as possible, as it has the most impact on cache-efficiency. The MatrixUCB algorithm is problematic in this regard, since it has to store $n^2$ statistics for the matrix instead of $2 n$  like most adversarial bandits, with respect to the number of actions $n$.
* `ChanceStats`
This storage is not needed by the provided algorithms, but it could be useful in the future. It's worth mentioning that if the struct is empty and its associated methods are no-ops, it should be entirely optimized away by the compiler.

## Concepts/Interface
### IsBanditAlgorithmTypes
```cpp
{
    bandit.get_empirical_strategies(matrix_stats, strategy, strategy)
} -> std::same_as<void>;
{
    bandit.get_empirical_value(matrix_stats, value)
} -> std::same_as<void>;
{
    bandit.get_refined_strategies(matrix_stats, strategy, strategy)
} -> std::same_as<void>;
{
    bandit.get_refined_value(matrix_stats, value)
} -> std::same_as<void>;
```
Used as the 'final answer' interface. All bandits can offer empirical value and policies for this purpose, and in most reinforcement learning schemes this is taken to be the training target. There are many cases where you may want to use other estimates instead, hence the 'refined' version.
e.g. AlphaZero stuff using `q` over `z`, or a weighted average of the two.
```cpp
{
    bandit.initialize_stats(0, state, model, matrix_stats)
} -> std::same_as<void>;
```
This method is likely to be removed in the future. 
The vanilla MatrixUCB algorithm assumes that we know the number of iterations in advance, which is only true for the root node. This method was used to set the total number of iterations in the `MatrixStats` of the root node. Any other node would use the expected iterations of its parent matrix node to estimate its own expected iterations. 
```cpp
{
    bandit.expand(state, matrix_stats, model_output)
} -> std::same_as<void>;
```
This method properly initializes the statistics of the matrix stats. In exp3 for example, it zero-initializes the gains and visit count vectors for both players.
```cpp
{
    bandit.select(device, matrix_stats, outcome)
} -> std::same_as<void>;
```
Chooses the joint actions for both players to commit and hence the associated chance node to visit.
```cpp
{
    bandit.update_matrix_stats(matrix_stats, outcome)
} -> std::same_as<void>;
{
    bandit.update_chance_stats(chance_stats, outcome)
} -> std::same_as<void>;
```
Updates the stats in the matrix and selected chance node for the next episode. Uses the value observed at the end of the forward phase.
```cpp
{
    outcome.row_idx
} -> std::same_as<int&>;
{
    outcome.col_idx
} -> std::same_as<int&>;
{
    outcome.value
} -> std::same_as<typename Types::Value &>;
```
A struct that is used for storage of any data or observations that need to be shared between the selection and update phases. For virtually all bandit algorithms, this will need to include the action (indices) that were selected. `Rand` does not store even these simply because its update is a no-op.
### IsMultithreadedBanditTypes
```cpp
{
    bandit.select(device, matrix_stats, outcome, mutex)
} -> std::same_as<void>;
{
    bandit.update_matrix_stats(matrix_stats, outcome, mutex)
} -> std::same_as<void>;
{
    bandit.update_chance_stats(chance_stats, outcome, mutex)
} -> std::same_as<void>;
```
The multi-threaded search needs to lock the mutex guarding the search stats when one thread is performing an update. It may be that the mutex can be unlocked *before* the method is finished, so we pass a reference to the mutex to allow for this. Otherwise, the entire update would be sandwiched between `lock()` and `unlock()` calls. Reducing contention is the most effective way to increase performance of multi-threaded algorithms.
### IsOffPolicyBanditTypes
```cpp
{
    bandit.update_matrix_stats(matrix_stats, outcome)
} -> std::same_as<void>;
{
    bandit.update_chance_stats(chance_stats, outcome)
} -> std::same_as<void>;
```

# TreeAlgorithm

The bandit algorithm decides how the tree is expanded and explored, which is outside the scope of the bandit algorithm.

**These classes are derived from the bandit algorithm.**

Because of this, the tree algorithm inherits all of the methods of the bandit algorithm, and can therefore use them it methods like `run_iteration`.

It will also inherit the three structs `MatrixStats`, `ChanceStats`, and `Outcome` from before. Inheriting the methods and data structures of the bandit algorithms is what allows the interchangeability of latter.

But the multi-threaded and off-policy tree algorithm will not use the `MatrixStats` and `ChanceStats` just as inherited. Instead, they will derive their own structs to add necessary data. For the multi-threaded tree algorithms, a mutex or mutex-id is added to protect the data in the stats from race conditions.

```cpp
struct MatrixStats : BanditAlgorithm::MatrixStats
{
    typename Types::Mutex mutex{};
};
struct ChanceStats : BanditAlgorithm::ChanceStats
{
    typename Types::Mutex mutex{};
};
```

The matrix and chance nodes are templates that accept algorithms as parameters. It is the *tree* algorithms that is passed as arguments for this, so that the augmented `TreeAlgorithm::MatrixStats` and `TreeAlgorithm::ChanceStats` are used in the tree structure, not their respective base classes.

## Concepts/Interface
```cpp
{
    search = search
} -> std::same_as<typename Types::Search &>;
{
    search.run(0, device, state, model, matrix_node)
} -> std::same_as<size_t>;
{
    search.run_for_iterations(0, device, state, model, matrix_node)
} -> std::same_as<size_t>;
{
    search.run(0, device, state, model, matrix_node)
} -> std::same_as<size_t>;
```

## Implementations

Unlike the bandit algorithms, it is expected that the provided tree algorithms should accommodate most experiments.

### TreeBandit
Essentially vanilla MCTS

### TreeBanditThreaded
The CRTP is used here to add a mutex member to the matrix stats of the bandit algorithm. This mutex is locked before accessing chance stats for selection and updating.

### TreeBanditThreadPool
To save on memory compared to the above, the instances of the algorithms maintain a pool of mutexes, and the index of a matrix node is stored in its stats instead.

### OffPolicy
The name might be misleading. Its basically intended for use with batched GPU inference.

The reason for the name is the way it handles selecting and updating nodes. Many other implementations of parallel search use 'virtual loss' as a way of diversifying the threads' leaf node selection. Additionally with this method we update the matrix node stats like normal during the backward phase.

This method relies on certain properties of UCB, the standard bandit algorithm in games like Chess. Instead we use artificial sampling (like tossing repeated leaf nodes) and instead treat the process of updating the stats in the backward phase as an instance of off policy learning in RL. This base maintains the 'actor policies' (`row_mu, col_mu`) of the forward phase and uses those in conjunction with the selection probabilities in the backward phase (the "learner" policy) to calculate the ratio `pi / mu`. This coefficient to adjust the 'learning rate' of the node is a common trick in order to un-bias the samples in the context of off-policy RL.
