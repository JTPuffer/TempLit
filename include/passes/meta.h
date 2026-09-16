#pragma once

#include <concepts>
#include <type_traits>

namespace passes {

template <typename... Ts> struct type_list {};

template <typename T> struct is_type_list : std::false_type {};

template <typename... Ts>
struct is_type_list<type_list<Ts...>> : std::true_type {};

template <typename T>
inline constexpr bool is_type_list_v = is_type_list<T>::value;

template <typename T>
concept TypeList = is_type_list_v<T>;

template <typename...> inline constexpr bool always_false_v = false;

template <typename... Lists> struct concat;

template <> struct concat<> {
  using type = type_list<>;
};

template <typename... Ts> struct concat<type_list<Ts...>> {
  using type = type_list<Ts...>;
};

template <typename... Lhs, typename... Rhs, typename... Rest>
struct concat<type_list<Lhs...>, type_list<Rhs...>, Rest...>
    : concat<type_list<Lhs..., Rhs...>, Rest...> {};

template <typename... Lists> using concat_t = typename concat<Lists...>::type;

template <typename List, typename T> struct contains_type;

template <typename... Ts, typename T>
struct contains_type<type_list<Ts...>, T>
    : std::bool_constant<(std::same_as<Ts, T> || ...)> {};

template <typename List, typename T>
inline constexpr bool contains_type_v = contains_type<List, T>::value;

template <typename List, typename T> struct append_unique;

template <typename... Ts, typename T>
struct append_unique<type_list<Ts...>, T> {
  using type = std::conditional_t<contains_type_v<type_list<Ts...>, T>,
                                  type_list<Ts...>, type_list<Ts..., T>>;
};

template <typename Input, typename Output = type_list<>> struct unique;

template <typename Output> struct unique<type_list<>, Output> {
  using type = Output;
};

template <typename Head, typename... Tail, typename Output>
struct unique<type_list<Head, Tail...>, Output>
    : unique<type_list<Tail...>,
             typename append_unique<Output, Head>::type> {};

template <typename List> using unique_t = typename unique<List>::type;

} // namespace passes
