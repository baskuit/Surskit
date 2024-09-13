#pragma once

#include <libpinyon/packs.h>

#include <tuple>

/*

search_type_generator<...>(
    TypePack<...> bandit_type_pack,
    TemplatePack<...> node_template_pack)

will return a tuple of `IsSearchTypes` type lists

*/

namespace detail
{
    template <
        template <typename, template <typename...> typename, typename...> typename TreeBanditTemplate,
        template <typename...> typename NodeTemplate,
        typename... BanditTypes>
    auto search_type_generator_unpack_bandits(TypePack<BanditTypes...>)
    {
        return std::make_tuple(TreeBanditTemplate<BanditTypes, NodeTemplate>{}...);
    }

    template <
        template <typename, template <typename...> typename, typename...> typename TreeBanditTemplate,
        typename BanditTypePack,
        template <typename...> typename... NodeTemplates>
    auto search_type_generator_unpack_nodes(BanditTypePack bandit_type_pack, NodeTemplatePack<NodeTemplates...>)
    {
        return std::tuple_cat((
            search_type_generator_unpack_bandits<TreeBanditTemplate, NodeTemplates>(bandit_type_pack))...);
    }
};

template <
    template <typename, template <typename...> typename, typename...> typename... TreeBanditTemplates,
    typename BanditTypePack,
    typename NodeTemplatePack>
auto search_type_generator(BanditTypePack bandit_type_pack, NodeTemplatePack node_template_pack)
{
    return std::tuple_cat((
        detail::search_type_generator_unpack_nodes<TreeBanditTemplates>(bandit_type_pack, node_template_pack))...);
}
