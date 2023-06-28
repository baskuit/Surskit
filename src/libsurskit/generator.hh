#include <tuple>
#include <type_traits>
#include <ranges>
#include <functional>

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

    CartesianProductGenerator(Function function, Containers &...args)
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