//
// Created by Dekai WU and YAN Yuchen on 20210119.
//

#ifndef LEGO_LEGO_PARAM_NAMING_GUARD_HPP
#define LEGO_LEGO_PARAM_NAMING_GUARD_HPP

#include <vector>
#include <string>
namespace tg {

  class optimizer_base;
  class backprop_trainable_parameter_base;

  /**
   * \brief When declaring transducer models, use this guard to append path to all created parameters.
   *
   * For example:
   *     transducer_model my_dense;
   *     {
   *       lego_param_naming_guard _("first_name");
   *       {
   *         lego_param_naming_guard _("second_name");
   *         my_dense = dense_structure.initialize(2,2);
   *       }
   *     }
   * In this case, my_dense will have all its internal parameters to have path first_name/second_name
   *
   */
  class lego_param_naming_guard {
    static thread_local std::vector<std::string> v;
    friend optimizer_base;
    friend backprop_trainable_parameter_base;
  public:

    lego_param_naming_guard(const lego_param_naming_guard&) = delete;
    lego_param_naming_guard(lego_param_naming_guard&&) noexcept = delete;
    lego_param_naming_guard& operator=(const lego_param_naming_guard&) = delete;
    lego_param_naming_guard& operator=(lego_param_naming_guard&&) noexcept = delete;

    lego_param_naming_guard(std::string name) {
      v.push_back(std::move(name));
    }

    ~lego_param_naming_guard() {
      v.pop_back();
    }
  };

  struct lego_param_path {
    std::vector<std::string> values;
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(values);
    }
    lego_param_path() = default;
    lego_param_path(const lego_param_path&) = default;
    lego_param_path(lego_param_path&&) noexcept = default;
    lego_param_path& operator=(const lego_param_path&) = default;
    lego_param_path& operator=(lego_param_path&&) noexcept = default;
    explicit lego_param_path(std::vector<std::string> values);
    bool contains(const std::vector<std::string>& fragment) const;
    bool starts_with(const std::vector<std::string>& fragment) const;
    bool ends_with(const std::vector<std::string>& fragment) const;
  };
}

#endif //LEGO_LEGO_PARAM_NAMING_GUARD_HPP
