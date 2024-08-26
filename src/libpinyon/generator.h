#pragma once

#include <tuple>
#include <type_traits>
#include <ranges>
#include <functional>

/*

Provides an object with a range interface. There are 3 main parameters

1. A Collection of containers
2. A type Output that is the return type of the iterator
3. A function from the cartesian product of the containers that returns type Output

Example 

int function (int a, int b) {
    return 10 * a + b;
}
auto generator<Output>({1, 2}, {3, 4});
auto results_vec = std::vector<int>{};

for (auto x : generator) {
    results_vec.push_back(x);
}

// results_vec = {13, 14, 23, 24}

*/

template <typename Output, typename... Containers>
struct CartesianProductGenerator
{
    std::tuple<Containers...> containers;
    using Tuple = std::tuple<typename Containers::value_type...>;
    using Product = decltype(std::apply([](auto &...containers)
                                        { return std::views::cartesian_product(containers...); },
                                        containers));
    using It = std::ranges::iterator_t<Product>;

    Product cart_prod_view = std::apply([](auto &...containers)
                                        { return std::views::cartesian_product(containers...); },
                                        containers);

    using Function = std::function<Output(Tuple)>;
    Function function;

    CartesianProductGenerator(Function function, Containers ...args)
        : function{function}, containers(std::forward<Containers>(args)...)
    {
    }

    class Iterator;

    Iterator begin()
    {
        return Iterator(cart_prod_view.begin(), this);
    }

    Iterator end()
    {
        return Iterator(cart_prod_view.end(), this);
    }

    class Iterator : public It
    {
    public:
        CartesianProductGenerator *ptr;

        Iterator(const It &it, CartesianProductGenerator *ptr) : It{it}, ptr{ptr}
        {
        }

        Iterator &operator++()
        {
            It::operator++();
            return (*this);
        }

        Output operator*()
        {
            Tuple tuple = It::operator*();
            return (ptr->function)(tuple);
        }

        bool operator==(const Iterator &other) const
        {
            return static_cast<const It &>(*this) == static_cast<const It &>(other);
        }
    };
};
