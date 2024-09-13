
# Solver

These algorithms perform the simultaneous-move analog of mini-max; They **solve** for Nash equilibrium payoffs and strategies on a sub-tree. The calculated Nash equilibrium payoffs and strategies are stored in the `MatrixStats`.

Solving requires that the game is constant-sum. 

The sub-tree is expanded up to `depth=-1`, meaning infinite depth by default. The payoffs at a 'terminal' node (terminal in the sub-tree, perhaps not in the entire game) are given by model inference. Since these algorithms are intended to be used in offline contexts, like deep analysis, the user may want to consider using a `SearchModel` for this purpose; This will provide stronger value estimation at the cost of speed.

These algorithms are run in a one-shot fashion, rather than online like tree-bandit. They cannot be terminated early, however they can still be run incrementally. The AlphaBeta algorithm will store a 'primary action' for both players that is used to start the algorithm. When the algorithm is called again on a previously solved node, but now with higher depth, it will use these actions to start the algorithm. This could reduce the number of explored nodes compared to solving once at the higher depth

# Concepts/Interface

```cpp
{
    search.run(0, device, state, model, matrix_node)
} -> std::same_as<size_t>;
```
The argument list is the same as the tree bandit searches. Instead of the number of iterations, the integral quantity in the first parameter denotes the depth to solve. The `PRNG` device is not used by the full traversal solver, but it may be used in the alpha beta solver when determining whether to stop exploring a chance node.

# Implementations

### FullTraversal

The simplest possible solver: expands the entire game tree with a single recursive function call and solves every matrix node. Because of this exhaustion, it produces a sub-game perfect solution to the game tree.

### AlphaBeta (aka SMAB)

This algorithm is an implementation of Bošanský, et al. (2013), modified for stochastic games. The paper optionally uses a sub-algorithm called 'Serialized AlphaBeta', which further tightens the bounds on the value of a matrix node. Serialized AlphaBeta is currently not implemented, and the paper admits it does not always improve performance, due to the increased tree traversal.

The paper should be the primary source on the exact workings of SMAB. Accommodating stochastic games is straightforward, and mirrors how 'Expectiminimax' extends the usual mini-max. 

There is a simple trick to improve vanilla SMAB for stochastic games. When we are calculating the entry `s_ij` in the payoff matrix, we are actually calculating the expected payoff of the chance node for that entry, which is simply calculating the expected value of all matrix nodes that chance node points to.
Since we know the chance actions and their associated probabilities, we can use that info and the MinVal, MaxVal to put bounds on the value of a chance node.
If `0 < p < 1` is the total probability of the *explored* chance actions, and the sum of the calculated the values is `v`, then value of the chance node `V` is bounded by

$$v + (1 - p) * \text{MinVal} < V < v + (1 - p) * \text{MaxVal}$$

This allows us to perform the feasibility check after each update to the value of `s_ij` instead of when its totally computed, to see if we can terminate early.

There is an implementation of stochastic SMAB that doesn't use this optimization and instead reflects the paper very closely.
