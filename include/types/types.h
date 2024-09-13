#pragma once

#include <concepts>

// sanity check that a type list has what's required
template <typename Types>
concept BasicTypeList = requires() {
  typename Types::Real;
  typename Types::template Vector<float>;
  typename Types::template Matrix<float>;
  typename Types::PRNG;
  typename Types::Mutex;
};