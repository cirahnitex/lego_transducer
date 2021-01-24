//
// Created by Dekai WU and YAN Yuchen on 20200627.
//

#include "include/lego_boolean_operations.hpp"
#include "boolean_operations_impl.hpp"
#include "transducer_variant.hpp"
#include "include/transducer_model.hpp"
using namespace tg;
using namespace std;

value_placeholder
tg::lazy_ifelse(const value_placeholder& cond, const value_placeholder& val_if_true, const value_placeholder& val_if_false) {
  static transducer_model model(make_shared<transducer_variant>(lazy_ifelse_op()));
  return model(cond, val_if_true, val_if_false);
}

value_placeholder tg::eager_ifelse(const tg::value_placeholder& cond, const tg::value_placeholder& val_if_true,
                                  const tg::value_placeholder& val_if_false) {
  static transducer_model model(make_shared<transducer_variant>(eager_ifelse_op()));
  return model(cond, val_if_true, val_if_false);
}

value_placeholder tg::soft_ifelse(const tg::value_placeholder& cond, const tg::value_placeholder& val_if_true,
                                 const tg::value_placeholder& val_if_false) {
  static transducer_model model(make_shared<transducer_variant>(soft_ifelse_op()));
  return model(cond, val_if_true, val_if_false);
}

value_placeholder tg::to_boolean(const value_placeholder& x) {
  static transducer_model model(make_shared<transducer_variant>(to_boolean_op()));
  return model(x);
}

namespace tg {
  value_placeholder operator!(const tg::value_placeholder& x) {
    static transducer_model model(make_shared<transducer_variant>(logical_not_op()));
    return model(x);
  }

  tg::value_placeholder operator==(const tg::value_placeholder& x, const tg::value_placeholder& y) {
    static transducer_model model(make_shared<transducer_variant>(eq_op()));
    return model(x, y);
  }

  tg::value_placeholder operator!=(const tg::value_placeholder& x, const tg::value_placeholder& y) {
    static transducer_model model(make_shared<transducer_variant>(ne_op()));
    return model(x, y);
  }

  tg::value_placeholder operator<(const tg::value_placeholder& x, const tg::value_placeholder& y) {
    static transducer_model model(make_shared<transducer_variant>(lt_op()));
    return model(x, y);
  }

  tg::value_placeholder operator>(const tg::value_placeholder& x, const tg::value_placeholder& y) {
    static transducer_model model(make_shared<transducer_variant>(gt_op()));
    return model(x, y);
  }

  tg::value_placeholder operator&&(const tg::value_placeholder& x, const tg::value_placeholder& y) {
    static transducer_model model(make_shared<transducer_variant>(logical_and_op()));
    return model(x, y);
  }

  tg::value_placeholder operator||(const tg::value_placeholder& x, const tg::value_placeholder& y) {
    static transducer_model model(make_shared<transducer_variant>(logical_or_op()));
    return model(x, y);
  }

}

