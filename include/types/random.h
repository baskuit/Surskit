#pragma once

#include <array>
#include <random>

class prng {
private:
  std::mt19937::result_type seed;
  std::mt19937 engine;
  std::uniform_real_distribution<double> uniform_;
  std::uniform_int_distribution<uint64_t> uniform_64_;

public:
  prng() : seed(std::random_device{}()), engine(std::mt19937{seed}) {}
  prng(std::mt19937::result_type seed)
      : seed(seed), engine(std::mt19937{seed}) {}

  std::mt19937::result_type get_seed() const noexcept { return seed; }

  std::mt19937::result_type random_seed() noexcept {
    return uniform_64_(engine);
  }

  // Uniform random in (0, 1)
  double uniform() noexcept { return uniform_(engine); }

  // Random integer in [0, n)
  int random_int(int n) noexcept { return uniform_64_(engine) % n; }

  uint64_t uniform_64() noexcept { return uniform_64_(engine); }

  template <typename Container>
  int sample_pdf(const Container &input) noexcept {
    double p = uniform();
    for (int i = 0; i < input.size(); ++i) {
      p -= static_cast<double>(input[i]);
      if (p <= 0) {
        return i;
      }
    }
    return 0;
  }

  template <template <typename...> typename Vector, typename T>
    requires(T::get_d())
  int sample_pdf(const Vector<T> &input) noexcept {
    double p = uniform();
    for (int i = 0; i < input.size(); ++i) {
      p -= input[i].get_d();
      if (p <= 0) {
        return i;
      }
    }
    return 0;
  }

  void discard(size_t n) { engine.discard(n); }
};
