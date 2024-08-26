
# Pinyon

Pinyon is a library for research and development of search and solving algorithms for (perfect information) simultaneous move, stochastic games. The code emphasizes:

### Compile-Time Optimization
This library uses the template metaprogramming features of C++ to eliminate the cost of game/alorithm/etc agnosticism of similar libraries like [alpha-zero-general](https://github.com/suragnair/alpha-zero-general). Pinyon is then fully capable of tasks like powering expensive reinforcement-learning projects or server analysis.

### Fully Modular
The game, evaluation function, and search algorithm are liberally interchangible. The interface was designed from the beginning to unify the major methods. Serveral other basic utilities are provided as well.

### Simple to Start With
Virtually any game can be augmented with a simple and fast monte-carlo search function, providing guarantees of convergence to Nash Equilibrium.

### Highly Abstractable
The interface uses a template convention that facilitates the layering of multiple utilities to trivialize the implementation of otherwise intricate scripts, exemplified by TODO ([pkmn-pinyon-test](https://github.com/baskuit/pkmn-pinyon-test) script to replicate q-value matrices of `bull.cc` via a depth=1 `FullTraversal` backed by two layer `AlphaBeta` + `Exp3` model at leaf nodes.)

### Easy Benchmarking 
Various *implementations* of the same object can be quickly and automatically benchmarked for cheaper compute. See `/benchmarking`.

### Minimal Code
Many tasks can be settled with a small script. Once a process is described in terms of the interface it's usually almost complete.

### Minimal Setup
This is a header library that only depends on the GNU Multiple Precision Library. Inclusion into projects is done via cmake, i.e. `add_directory(extern/pinyon)` and `#include <pinyon.h>`. 

# Documentation
The library is extensively documented and the points above are expanded therein. The organization of the abstraction/modularity layers is reflected by the directory structure.

The interface is codified with the use of C++20 `Concepts` which serve as another for of documentation.

# Installation
The tests and benchmarking tools can be built via
```
sudo apt install libgmp3-dev
git clone --recurse-submodules https://github.com/baskuit/pinyon
cd pinyon
mkdir build
cd build
cmake ..
make
```

# Testing
The library comes with several tests to check the correctness of the algorothm implementations, among other things. To run, simply uncomment the `#tests` portion of the CMakeLists; It is a script that builds everything in `/tests` and marks it as such. Then simply run
```
cd build
ctest tests
```
