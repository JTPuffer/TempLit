#pragma once

#include "passes/meta.h"
#include "passes/program.h"

namespace passes {

template <typename Program> struct get_local_gradient_paths;

template <typename... Ops>
struct get_local_gradient_paths<operations<Ops...>> {
  using type = concat_t<typename get_local_gradient_paths<Ops>::type...>;
};

template <typename PROG1, typename PROG2>
struct get_local_gradient_paths<prog_link<PROG1, PROG2>> {
  using type = concat_t<typename get_local_gradient_paths<PROG1>::type,
                        typename get_local_gradient_paths<PROG2>::type>;
};

template <typename Grad, typename Path>
struct get_local_gradient_paths<accumulate_local_grad<Grad, Path>> {
  using type = type_list<Path>;
};

template <typename Grad, typename Path>
struct get_local_gradient_paths<accumulate_grad<Grad, Path>> {
  using type = type_list<>;
};

template <> struct get_local_gradient_paths<no_grad> {
  using type = type_list<>;
};

} // namespace passes
