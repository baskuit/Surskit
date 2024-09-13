
# Model
Any search or solving method requires a way to estimate the Nash equilibrium values of a given state. A model is anything that provides this value estimate along with any other information needed for search, like Policy estimation.
Borrowing from machine learning, the process of estimating information using a model is called 'inference'.
The value estimate should represent the expected score of a player starting from that state. Usually games assign a payoff of $0$ for a loss, $\frac{1}{2}$ for a draw, and $1$ for a win. In this value for a player should be in $[0, 1]$. Old-school evaluations like material advantage in Chess (e.g. "+3" is a winning advantage) are not appropriate and do not meet some of the basic assumption for convergence.

### Resource Management
Models are assumed to be copy constructable. Neural network based models use shared pointers to the network, and heuristic and test models should have little to no data to copy.

### Symmetry
Models make the usual assumption that there is a fixed 'row' and 'column' player, but neither perspective is privileged. Since we also assume perfect information, there is no reason that a model should provide information for one player and not the other.
Therefore the value estimate of a model applies to both players. More precisely, the estimate is of `Value` type, with methods `get_row_payoff()` and `get_col_payoff()` returning a `Real` type.
### Policy
Not all search algorithms require a policy estimate for the players, and many model implementations do not provide one during inference. The Monte-Carlo model has a template parameter that enables or disables the policy output.

### I/O
The interface for models is designed with the most important examples in mind: Monte-Carlo and Libtorch. The Monte-Carlo inference will 'consume' a state that it inferences because of the rollout process. All search algorithms are fine with this behavior and don't require the state to be copied, so we assume that all models will consume the states they inference. This eliminates unneeded copying.

### Batched I/O
The off-policy algorithm for batched inference requires a type that corresponds to the tensor input and tensor output of a GPU based model. There's no reason that this search algorithm should not work for non-GPU based models, so we define `ModelBatchInput` and `ModelBatchOutput`.
For non-tensor based models, then these types are usually just `std::vector<typename Types::State>` and `std::vector<typename Types::ModelOutput>`. The batched `inference(&ModelBatchInput, &ModelBatchOutput)` method will just call the normal `inference(State&&, ModelOutput &)` on the pairs of vector elements. 

### `ModelBandit` and `SearchModel`
There is a utility called ModelBandit that evaluates the strength of a fixed pool of 'agents' by having two agents play out games from the beginning and returning the average payoff for each agent.
These 'agents' are intended to be bandit algorithms that a run for some amount of iterations/time, as a way to evaluate the strength of different algorithms. attached models, and hyper parameters.
However, we might also want to treat the unrefined policy output of a model as an agent too. Examples include the empty model, with uniform policy output, and the solved model.

# Concepts/Interface

### IsValueModelTypes
```cpp
{
    output.value
} -> std::same_as<typename Types::Value &>;
```
The value estimate is a public member of the `ModelOutput` type.
```cpp
{
    model.inference(std::forward<typename Types::State>(moved_state), output)
} -> std::same_as<void>;
```
This concept means that the inference method takes an rvalue reference to a state as its first parameter. We usually invoke `std::move` when calling inference, otherwise we create a copy directly in the argument `inference(typename Types::State{state}, ...`
### IsBatchModelTypes
```cpp
{
    model.inference(model_batch_input, model_batch_output)
} -> std::same_as<void>;
{
    model.add_to_batch_input(std::forward<typename Types::State>(moved_state), model_batch_input)
} -> std::same_as<void>;
{
    model_batch_input[0]
} -> std::same_as<typename Types::ModelInput &>;
{
    model_batch_output[0]
} -> std::same_as<typename Types::ModelOutput &>;
```
TODO

### IsPolicyModelTypes
```cpp
{
    output.row_policy
} -> std::same_as<typename Types::VectorReal &>;
{
    output.col_policy
} -> std::same_as<typename Types::VectorReal &>;
```

# Implementations

### MonteCarloModel

This model applies the standard Monte-Carlo rollout method of value estimation. It is universal in the sense that it can be applied to any `PerfectInfoState`.

Monte-Carlo inference is very weak. It is outperformed by any decent heuristic model. It's strength is also highly dependent on "depth-to-terminal"; The farther away a game state is from conclusion, the weaker this model's inference is. The same can be said for its computational performance as well for obvious reasons.

There are no limits on the length of a rollout. If a state (read: battle) becomes corrupted, typically by improper initialization or committing invalid actions, then it is possible that the corrupted state won't throw an exception or cause a run-time error, but it will also never reach terminality. This causes the Monte Carlo inference to hang indefinitely.

### Libtorch

### SolvedModel
This model assumes the underlying state type is a solved state and it simply uses the Nash equilibrium strategies and payoffs returned by `get_strategies` and `payoff` as its output. It uses the same batched I/O types as the Monte-Carlo model.

### SearchModel
The bandit algorithms are expected to have methods `get_empirical_strategies`, `get_emprical_values`, and these can be treated as policy and value estimates

One immediate application of this is for the FullTraversal and AlphaBeta solvers, where we can use this 'wrapped search' as the model. If these solvers are run with a finite max_depth, then this model will perform a normal bandit search at the leaf nodes of the sub-tree. 
This creates a 'hybrid' search algorithm that could reap the rewards of both solving and tree bandit search styles. 

### NullModel
This model gives $\frac{1}{2}, \frac{1}{2}$ as the value estimate and the uniform distribution over the actions as the policy estimate. It is used for benchmarking   
