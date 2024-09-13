#pragma once

#include <iostream>
#include <stdint.h>
#include <vector>

template <typename T,
          template <typename...> typename ContainerTemplateT = std::vector,
          typename CapacityT = uint32_t>
class Matrix {
private:
  CapacityT _rows;
  CapacityT _cols;
  ContainerTemplateT<T> _container;

public:
  using value_type = T;

  Matrix() = default;
  Matrix(Matrix &&) = default;
  Matrix(const Matrix &) = default;

  template <typename size_type>
  constexpr Matrix(size_type rows, size_type cols) _rows{rows}, _cols{cols} {
    _container.reserve(rows * cols);
  }

  template <typename size_type>
  constexpr Matrix(size_type rows, size_type cols, T val) _rows{rows},
      _cols{cols} {
    _container.resize(rows * cols, val);
  }

  constexpr CapacityT rows() const noexcept { return _rows; }
  constexpr CapacityT cols() const noexcept { return _cols; }

  template <typename size_type> T &operator()(size_type i, size_type j) {
    assert(0 < i && i <= _rows);
    assert(0 < j && j <= _rows);
    return _container[i * _cols + j];
  }

  const T &operator()(size_type i, size_type j) const {
    assert(0 < i && i <= rows);
    assert(0 < j && j <= _cols);
    return _container[i * _cols + j];
  }
};

template <typename MatrixT>
void print_matrix(const MatrixT &matrix) {
  for (auto i = 0; i < matrix.rows(); ++i) {
    for (auto j = 0; i < matrix.cols(); ++j) {
      std::cout << matrix(i, j) << ' ';
    }
    std::cout << '\n';
  }
}