#pragma once

/*

Store m, n actions as 8 bit each. 

Simplest is just encode as 1/256 fractions, and omit the last action, since its prob is determined

*/

#include <limits>

template <typename Int = uint8_t>
struct Strategy : public std::vector<Int> {

    static constexpr size_ = std::numeric_limits<int>::max();
    // some kind of unsigned int. static assert?
    // this basically determines how precise this representation is
    
    template <typename T>
    T operator[](size_t index) {
        return static_cast<>((*this)[index]) / static_cast<T>(size_);
    }

    void sort () {
        // calculate redidual prob, if larger than least replace value
        // [a, c] -> b > c -> [a, b]
    }

    template <typename T>
    void sort (std::vector<T> &companion) {
        // also sorted
    }

    // faster when sorted
    template <class PRNG>
    sample (PRNG& device) {
        Int choice = device.get<Int>();

        while (choice > 0) {
            break;
            // subtract stored Int sequentially
        }
    }    
};

template <typename Int>
struct AliasTable {

    // https://en.wikipedia.org/wiki/Alias_method

    /*
    Has to be computed from distro, and vice versa.
    Otherwise samples in one shot using only (0, 1) or Int / MAXINT

    but it does require two times the storage, so idk
    
    We dont really frequently sample from static distros though.
    */

   explicit AliasTable(const Strategy<Int> &s);

};