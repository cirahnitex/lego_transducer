//
// Created by Dekai WU and YAN Yuchen on 20200714.
//
/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_CLASS_TRAITS_HELPER_CMACRO_HPP
#define LEGO_CLASS_TRAITS_HELPER_CMACRO_HPP

#define STATIC_MEMBER_DETECTOR(detector_name, member_name) \
template <typename U> \
class detector_name \
{ \
private: \
    template<typename T> struct helper; \
    template<typename T> \
    static constexpr bool check(helper<decltype(T::member_name)>*) {return true;} \
    template<typename T> \
    static constexpr bool check(...) {return false;} \
public: \
    static constexpr bool value = check<U>(0); \
}; \
template <typename T> \
inline constexpr bool detector_name##_v = detector_name<T>::value; \

#define NULLARY_MEMBER_FUNCTION_DETECTOR(detector_name, member_name, ret_type) \
template<typename U> \
class detector_name { \
  template <typename T, T> struct helper; \
  template <typename T> \
  static constexpr bool check(helper<ret_type (T::*)(), &T::member_name>*) { \
    return true; \
  } \
  template <typename T> \
  static constexpr bool check(...) { \
    return false; \
  } \
public: \
  static constexpr bool value = check<U>(nullptr); \
}; \
template <typename T> \
inline constexpr bool detector_name##_v = detector_name<T>::value; \

#define MEMBER_FUNCTION_DETECTOR(detector_name, member_name, ret_type, ...) \
template<typename U> \
class detector_name { \
  template <typename T, T> struct helper; \
  template <typename T> \
  static constexpr bool check(helper<ret_type (T::*)(__VA_ARGS__), &T::member_name>*) { \
    return true; \
  } \
  template <typename T> \
  static constexpr bool check(...) { \
    return false; \
  } \
public: \
  static constexpr bool value = check<U>(nullptr); \
}; \
template <typename T> \
inline constexpr bool detector_name##_v = detector_name<T>::value; \

#endif //LEGO_CLASS_TRAITS_HELPER_CMACRO_HPP
