#include <types/types.h>
#include <types/matrix.h>

#include <mutex>

struct State {

};

struct Types {
    using Real = float;
    template <typename T>
    using Vector = std::vector<T>;
    template <typename T>
    using Matrix = Matrix<T>;
    using PRNG = void;
    using Mutex = std::mutex;
};

int main () {
    static_assert(BasicTypeList<Types>);

    return 0;
}