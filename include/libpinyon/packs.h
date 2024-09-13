#pragma once

#include <state/state.h>

template <template <typename...> typename... Template>
struct TemplatePack
{
};

template <
    template <
        typename...>
    typename... NodeTemplate>
struct NodeTemplatePack
{
};

template <typename... Type>
struct TypePack
{
};


template <typename... Ts, typename... Us, size_t... Is>
auto zip(const std::tuple<Ts...>& tuple1, const std::tuple<Us...>& tuple2, std::index_sequence<Is...>) {
    return std::make_tuple(std::make_pair(std::get<Is>(tuple1), std::get<Is>(tuple2))...);
}
template <typename... Ts, typename... Us>
auto zip(const std::tuple<Ts...>& tuple1, const std::tuple<Us...>& tuple2) {
    return zip(tuple1, tuple2, std::index_sequence_for<Ts...>());
}
