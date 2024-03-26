#pragma once

#include <cassert>
#include <functional>
#include <iostream>
#include <ranges>
#include <tuple>
#include <variant>

namespace gEng {

// Class designed to unwrap and print some common patterns:
// - uses specified function to print
// - uses print method if exists
// - uses operator<< if possible
// - dereference pointer if not null and prints dereferenced type
// - unwraps std::optional if not null prints nested type
// - unwraps range and prints each element.
// Usage:: OutWrapper(outs(), PrintFuncIfNecessary) << Obj1 << Obj2 ...;
template <typename OutTy, typename... Funcs> class OutWrapper {
  // Object properties testers.
  // Early concepts.
  template <typename T, typename = void>
  struct has_output_operator : std::false_type {};
  template <typename T>
  struct has_output_operator<
      T, std::void_t<decltype(std::declval<OutTy &>() << std::declval<T>())>>
      : std::true_type {};
  template <typename T>
  static constexpr bool has_output_operator_v = has_output_operator<T>::value;

  template <typename T, typename = void>
  struct has_print_method : std::false_type {};
  template <typename T>
  struct has_print_method<T, std::void_t<decltype(std::declval<T>().print(
                                 std::declval<OutTy &>()))>> : std::true_type {
  };
  template <typename T>
  static constexpr bool has_print_method_v = has_print_method<T>::value;

  template <typename T> struct is_optional_type : std::false_type {};
  template <typename T>
  struct is_optional_type<std::optional<T>> : std::true_type {};
  template <typename T>
  static constexpr bool is_optional_type_v = is_optional_type<T>::value;

  template <typename, typename = std::void_t<>>
  struct is_range_type : std::false_type {};

  template <typename T>
  struct is_range_type<T, std::void_t<decltype(std::declval<T>().begin()),
                                      decltype(std::declval<T>().end())>>
      : std::true_type {};
  template <typename T>
  static constexpr bool is_range_type_v = is_range_type<T>::value;

  // Storage type for functions.
  using FnContainer = decltype(std::forward_as_tuple(
      std::forward<Funcs>(std::declval<Funcs>())...));

  // No proper function is found.
  // End recursion and return std::nullopt
  template <typename Obj, typename T>
  static constexpr auto findCallable() -> std::optional<T> {
    return std::nullopt;
  }

  // Check if function placed in Idx position is able to
  // print Obj.
  template <typename Obj, typename T, T Idx, T... Indices>
  static constexpr auto findCallable() -> std::optional<T> {
    if constexpr (std::is_invocable_v<std::tuple_element_t<Idx, FnContainer>,
                                      Obj, OutTy &>)
      return Idx;
    else
      return findCallable<Obj, T, Indices...>();
  }

  // Go through all functions recorded in tuple and find
  // the first which could print Obj.
  template <typename Obj, typename T, T... Indices>
  static constexpr auto
  findCallable(std::integer_sequence<T, Indices...>) -> std::optional<T> {
    return findCallable<Obj, T, Indices...>();
  }

  // Find an index of function in Calls, which can print Obj.
  template <typename ObjType> static constexpr auto getFunctionForObj() {
    return findCallable<ObjType>(
        std::make_integer_sequence<std::size_t,
                                   std::tuple_size_v<FnContainer>>{});
  }

  // Construct callable function to print Obj depending on Obj type
  // and properties.
  template <typename ObjTy> auto constructFunctionForObj() const {
    using ObjType = std::remove_cv_t<std::remove_reference_t<ObjTy>>;

    if constexpr (constexpr auto X = getFunctionForObj<ObjTy>()) {
      return std::ref(std::get<*X>(Calls));
    } else if constexpr (std::is_pointer_v<ObjType> &&
                         !std::is_same_v<const char *, ObjType>) {
      auto &&Func = constructFunctionForObj<decltype(*std::declval<ObjTy>())>();
      return [Func](const ObjTy &Obj, OutTy &Out) {
        if (Obj)
          std::invoke(Func, *Obj, Out);
        else
          Out << "Null";
      };
    } else if constexpr (is_optional_type_v<ObjType>) {
      using SubTy = typename ObjType::value_type;
      auto &&Func = constructFunctionForObj<SubTy>();
      return [Func](const ObjTy &Obj, OutTy &Out) {
        if (Obj)
          std::invoke(Func, Obj.value(), Out);
        else
          Out << "None";
      };
    } else if constexpr (is_range_type_v<ObjType>) {
      using SubTy = typename std::iterator_traits<decltype(std::begin(
          std::declval<ObjType>()))>::value_type;
      auto &&Func = constructFunctionForObj<SubTy>();
      return [Func](const ObjTy &Obj, OutTy &Out) {
        if (Obj.empty())
          return;

        // FIXME this can be done via ranges, but
        // compiler do not support this yet.
        auto BegIt = std::begin(Obj);
        std::invoke(Func, *BegIt, Out);

        auto EndIt = std::end(Obj);
        while (++BegIt != EndIt) {
          Out << ' ';
          std::invoke(Func, *BegIt, Out);
        }
      };
    } else if constexpr (has_print_method_v<ObjTy>) {
      return [](const ObjTy &Obj, OutTy &Out) { Obj.print(Out); };
    } else if constexpr (has_output_operator_v<ObjTy>) {
      return [](const ObjTy &Obj, OutTy &Out) { Out << Obj; };
    } else {
      return [](const ObjTy &, OutTy &Out) {
        assert(false && "Cannot find proper function to print object\n");
      };
    }
  }

  OutTy &Out;
  FnContainer Calls;

public:
  OutWrapper(OutTy &Out, Funcs &&...Fs)
      : Out{Out}, Calls(std::forward_as_tuple(std::forward<Funcs>(Fs)...)) {}

  template <typename ObjTy> const OutWrapper &operator<<(ObjTy &&Obj) const {
    auto &&Call = constructFunctionForObj<ObjTy>();
    std::invoke(Call, std::forward<ObjTy>(Obj), Out);
    return *this;
  }

  OutWrapper(const OutWrapper &) = delete;
  OutWrapper(OutWrapper &&) = delete;
  OutWrapper &operator=(const OutWrapper &) = delete;
  OutWrapper &operator=(OutWrapper &&) = delete;

  virtual ~OutWrapper() = default;
};

// Calls OutWrapper not with any OutTy, but with std::ostream.
// out(outs(), FuncsIfNecessary) << Obj1 << Obj2 ...;
template <typename... Funcs> inline auto out(std::ostream &Out, Funcs &&...Fs) {
  return OutWrapper<std::ostream, Funcs...>{Out, std::forward<Funcs>(Fs)...};
}

#if 0
// Expample of ConditionalOut implementation.
class ConditionalOut final {
  static bool &DoPr;

  template <typename... Funcs>
  struct ThisOutWrapper : public OutWrapper<std::ostream, Funcs...> {
    ThisOutWrapper(std::ostream &Out, Funcs &&...Fs)
        : OutWrapper<std::ostream, Funcs...>{Out, std::forward<Funcs>(Fs)...} {}

    template <typename ObjTy>
    const ThisOutWrapper &operator<<(ObjTy &&Obj) const {
      if (DoPr)
        OutWrapper<std::ostream, Funcs...>::operator<<(
            std::forward<ObjTy>(Obj));
      return *this;
    }
  };

public:
  template <typename... Funcs>
  static auto out(std::ostream &Out, Funcs &&...Fs) {
    return ThisOutWrapper{Out, std::forward<Funcs>(Fs)...};
  }
};
#endif

} // namespace gEng
