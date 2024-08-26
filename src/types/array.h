#pragma once

#include <array>
#include <algorithm>

template <size_t MaxSize>
struct A
{
    template <typename T>
    struct Array : std::array<T, MaxSize>
    {
        size_t _size = 0;

        Array() {}

        Array (const size_t n) {
            _size = n;
            std::fill(this->begin(), this->end(), T{});
        }

        Array(const Array &other)
        {
            _size = other._size;
            std::copy(other.begin(), other.end(), this->begin());
        }

        Array &operator=(const Array &other)
        {
            _size = other._size;
            std::copy(other.begin(), other.end(), this->begin());
            return *this;
        }

        bool operator==(const Array &other) const
        {
            for (int i = 0; i < _size; ++i) {
                if ((*this)[i] != other[i]) {
                    return false;
                }
            }
            return _size == other._size;
        }

        void resize(size_t n, T value)
        {
            _size = n;
            std::fill(this->begin(), this->end(), value);
        }

        void resize(size_t n)
        {
            _size = n;
        }

        size_t size() const
        {
            return _size;
        }

        void clear () {
            _size = 0;
        }

        std::array<T, MaxSize>::iterator end()
        {
            return std::array<T, MaxSize>::begin() + _size;
        }

        const std::array<T, MaxSize>::const_iterator end() const
        {
            return std::array<T, MaxSize>::begin() + _size;
        }
    };
};
