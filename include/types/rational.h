#pragma once

#include "gmpxx.h"

#include <iostream>

template <typename T = int> struct Rational {

  T p = 1;
  T q = 1;

  void canonicalize() {
    // eulcid's algorithm
    T a = p;
    T b = q;
    T c;
    while (b) {
      a %= b;
      c = a;
      a = b;
      b = c;
    }
    p /= a;
    q /= a;
  }

  constexpr Rational() {}

  constexpr Rational(const T p) : p{p}, q{1} {}

  constexpr Rational(const T p, const T q) : p{p}, q{q} {}

  // constexpr Rational(size_t p, size_t q) : p{p}, q{q} {}

  bool operator==(const Rational &y) const { return (p == y.p) && (q == y.q); }

  constexpr Rational operator+(const Rational y) const {
    Rational z = {p * y.q + y.p * q, q * y.q};
    return z;
  }

  constexpr Rational operator*(const Rational y) const {
    Rational z{p * y.p, q * y.q};
    return z;
  }

  constexpr Rational operator/(const Rational y) const {
    Rational z{p * y.q, q * y.p};
    return z;
  }

  constexpr bool operator<(const Rational y) const { return p * y.q < y.p * q; }

  constexpr bool operator<=(Rational y) const { return p * y.q <= y.p * q; }

  constexpr bool operator>(Rational y) const { return p * y.q > y.p * q; }

  constexpr bool operator>=(Rational y) const { return p * y.q >= y.p * q; }

  constexpr Rational &operator+=(Rational y) {
    p = p * y.q + y.p * q;
    q = q * y.q;
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os, const Rational &x) {
    os << x.p << '/' << x.q;
    return os;
  }

  constexpr operator float() const { return p / (float)q; }

  constexpr operator double() const { return p / (double)q; }

  operator mpq_class() const { return mpq_class{p, q}; }
};
