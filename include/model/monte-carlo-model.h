#pragma once

template <typename PRNG, typename State, size_t trials = 1>
class MonteCarloModel {
private:
  PRNG device;

public:
  MonteCarloModel() : device{} {}
  MonteCarloModel(const PRNG &device) : device{device} {}

  const auto inference(State &&state) {
    const auto rollout = [](PRNG &device, State &state) {
      return state.payoff();
    };
    decltype(state.payoff()) value_total{};
    if constexpr (trials > 1) {
      for (size_t t = 0; t < trials - 1; ++t) {
        auto state_copy{state};
        value_total += rollout(device, state_copy);
      }
    }
    value_total += rollout(device, state);
    return value_total / trials;
  }
};
