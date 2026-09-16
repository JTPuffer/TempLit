#pragma once

#include <cstddef>

namespace passes {

struct graph_root {};
template <typename Parent, std::size_t Index> struct graph_path {};

struct prog_root {};
template <typename Parent, std::size_t Index> struct prog_path {};
template <typename... Ops> struct operations {};
template <typename Path> struct fwd_ref {};
template <typename Path> struct fwd_shape {};
template <typename Path> struct fwd_last_dim {};
template <typename Path> struct input_ref {};
template <typename Path> struct index_ref {};
template <typename EXPR, typename PROG_PATH> struct temp_ref {};
template <typename Expr, typename Path> struct fwd_save {};
template <typename Parent_path, std::size_t Index = 0>
struct local_prog_path {};
template <typename prog1, typename prog2> struct prog_link {};
template <typename Grad, typename Path> struct accumulate_grad {};
template <typename Grad, typename Path> struct accumulate_local_grad {};
template <typename Path> struct grad_ref {};

struct grad_seed {};
struct no_grad {};

} // namespace passes
