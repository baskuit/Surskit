
# Tree Structure
Because the tree structure is intended to represent *stochastic* games, we require two different kinds of nodes to faithfully represent the game tree: `MatrixNodes` and `ChanceNodes`.
The node implementations differ in how much information they store (debug stores bidirectional pointers) and how they store and access the pointers of their children.

### Ownership
A node owns its children in the sense that a nodes destructor will always delete all of the nodes children.
The children are stored as raw pointers instead of unique pointers currently. This means that if a node is default copied, it will also have ownership. For this reason the copy constructors of matrix and chance nodes are `deleted`.

### Matrix Node Primacy
Conceptually a matrix node corresponds to a state. A chance node is transitional since it represents the decision point of the 'chance player', an after-state.
Once consequence of this is that chance nodes are virtually never constructed by the user, instead they are constructed as a consequence of search operations.

### Pointer vs Reference
The top-level interface tends to pass references to (matrix) nodes while implementation functions use pointers. This is related to the point regarding ownership. The root node for a search is usually created in block scope instead of a heap allocated, and the ownership scheme means that the tree will be cleaned up when the root is destroyed at the end of scope. 
This means it is natural to pass a reference at the top level interface. Implementation functions tend to use pointers since nodes store pointers to their children and siblings.

# Concepts/Interface

## MatrixNode
```cpp
{
    matrix_node.stats
} -> std::same_as<typename Types::MatrixStats &>;
```
The search data member is accessed directly instead of by method. The data is not initialized when the node is constructed; That is expected to happen in the `expand`method  (or `expand_using_output` for "off-policy" search.)
```cpp
{
    matrix_node.access(0, 0)
} -> std::same_as<typename Types::ChanceNode *>;
```
This is a state agnostic way of accessing the child node corresponding to joint actions, assuming that action order is unique. It will return a pointer to the node, and if the node does not already exist then it will heap allocate it.
```cpp
{
    const_matrix_node.access(0, 0)
} -> std::same_as<const typename Types::ChanceNode *>;
```
This behaves the same except it will return `nullptr` rather than allocate. It is used in the `TraversedState::State` class so that the matrix node that identifies the state in the tree can be `const`.
```cpp
{
    const_matrix_node.is_expanded()
} -> std::same_as<bool>;
{
    matrix_node.expand(0, 0)
} -> std::same_as<void>;
```
This method marks a node as expanded and properly initializes all the data in a matrix node that is not stored in the `stats` member (that is done by the particular search algorithm).
The arguments are the number or row and column player actions. This is the only information needed from the state. In fact, for all node implementations besides `FlatNodes`, this method will *only* mark the node as expanded. For flat nodes, it will also allocate the heap vector of pointers to chance node children.
:warning: Note: the multi-threaded search assumes that `is_expanded()` is atomic, which is not always true. It currently is on x86 since read/write of cache aligned primitives is atomic on that architecture.
```cpp
{
    const_matrix_node.is_terminal()
} -> std::same_as<bool>;
{
    matrix_node.set_terminal()
} -> std::same_as<void>;
```
This is how the tree bandit search checks if a node is terminal. Solver methods do not use this function since nodes are not visited more than once, and the accompanying state has that information.

In all implementations, the methods `is_terminal` and `is_expanded` are mapped to `bool` members `terminal` and `expanded`. It may be profitable to use something smaller (e.g. a single byte) to store this info so the interface specifies getter and setter methods instead of direct access. 

## ChanceNode
```cpp
{
    chance_node.stats
} -> std::same_as<typename Types::ChanceStats &>;
{
    chance_node.access(obs)
} -> std::same_as<typename Types::MatrixNode *>;
{
    const_chance_node.access(obs)
} -> std::same_as<const typename Types::MatrixNode *>;
```
See their mirrors for matrix nodes.
