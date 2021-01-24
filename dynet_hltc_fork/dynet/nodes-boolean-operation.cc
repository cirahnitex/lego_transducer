//
// Created by Dekai WU and YAN Yuchen on 20200630.
//

#include "nodes-boolean-operation.h"
#include "dynet/tensor-eigen.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/functors.h"

#include "dynet/simd-functors.h"

using namespace std;

#define BVEC_BROADCAST_TO_DIM(bvec, from_dim, to_dim) bvec.broadcast( Eigen::array<unsigned, 2>{to_dim.batch_size() / from_dim.batch_size(), to_dim.batch_elems() / from_dim.batch_elems()})

namespace dynet {


  // ************* ToBoolean *************
#ifndef __CUDACC__

  string ToBoolean::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "boolean(" << arg_names[0] << ')';
    return s.str();
  }

  Dim ToBoolean::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in ToBoolean")
    return xs[0];
  }

#endif

  template<class MyDevice>
  void ToBoolean::forward_dev_impl(const MyDevice& dev, const vector<const Tensor *>& xs, Tensor& fx) const {
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]).cast<bool>().cast<float>();
  }

  template<class MyDevice>
  void ToBoolean::backward_dev_impl(const MyDevice& dev,
                                    const vector<const Tensor *>& xs,
                                    const Tensor& fx,
                                    const Tensor& dEdf,
                                    unsigned i,
                                    Tensor& dEdxi) const {
  }

  DYNET_NODE_INST_DEV_IMPL(ToBoolean)

  // ************* ConditionalExpression *************
#ifndef __CUDACC__

  string ConditionalExpression::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "ConditionalExpression(" << arg_names[0] << "," << arg_names[1] << "," << arg_names[2] << ')';
    return s.str();
  }

  Dim ConditionalExpression::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 3, "Failed input count check in ConditionalExpression")
    return xs[1];
  }

#endif
  
  template<class MyDevice>
  void
  ConditionalExpression::forward_dev_impl(const MyDevice& dev, const vector<const Tensor *>& xs, Tensor& fx) const {

    auto&& cond = tbvec(*xs[0]);
    auto&& cond_dim = xs[0]->d;
    auto&& mask = BVEC_BROADCAST_TO_DIM(cond, cond_dim, fx.d);
    auto&& val_if_true = tbvec(*xs[1]);
    auto&& val_if_false = tbvec(*xs[2]);
    tbvec(fx).device(*dev.edevice) = mask.select(val_if_true, val_if_false);
  }

  template<class MyDevice>
  void ConditionalExpression::backward_dev_impl(const MyDevice& dev,
                                                const vector<const Tensor *>& xs,
                                                const Tensor& fx,
                                                const Tensor& dEdf,
                                                unsigned i,
                                                Tensor& dEdxi) const {
    if (i == 1) {
      auto&& cond = tbvec(*xs[0]);
      auto&& cond_dim = xs[0]->d;
      auto&& mask = BVEC_BROADCAST_TO_DIM(cond, cond_dim, fx.d);
      tbvec(dEdxi).device(*dev.edevice) += mask * tbvec(dEdf);
    }
    else if (i == 2) {
      auto&& cond = tbvec(*xs[0]);
      auto&& cond_dim = xs[0]->d;
      auto&& mask = BVEC_BROADCAST_TO_DIM((1 - cond), cond_dim, fx.d);
      tbvec(dEdxi).device(*dev.edevice) += mask * tbvec(dEdf);
    }
  }

  DYNET_NODE_INST_DEV_IMPL(ConditionalExpression)


  // ************* LogicalNot *************
#ifndef __CUDACC__

  string LogicalNot::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "LogicalNot(" << arg_names[0] << ')';
    return s.str();
  }

  Dim LogicalNot::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in LogicalNot")
    return xs[0];
  }

#endif

  template<class MyDevice>
  void LogicalNot::forward_dev_impl(const MyDevice& dev, const vector<const Tensor *>& xs, Tensor& fx) const {
    tvec(fx).device(*dev.edevice) = 1 - tvec(*xs[0]);
  }

  template<class MyDevice>
  void LogicalNot::backward_dev_impl(const MyDevice& dev,
                                    const vector<const Tensor *>& xs,
                                    const Tensor& fx,
                                    const Tensor& dEdf,
                                    unsigned i,
                                    Tensor& dEdxi) const {
  }

  DYNET_NODE_INST_DEV_IMPL(LogicalNot)


  // ************* LogicalAnd *************
#ifndef __CUDACC__

  string LogicalAnd::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "LogicalAnd(" << arg_names[0] << "," << arg_names[1] << ')';
    return s.str();
  }

  Dim LogicalAnd::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in LogicalAnd")
    return xs[0].sum_dims() >= xs[1].sum_dims() ? xs[0] : xs[1];
  }

#endif

  template<class MyDevice>
  void LogicalAnd::forward_dev_impl(const MyDevice& dev, const vector<const Tensor *>& xs, Tensor& fx) const {
    auto&& d0 = xs[0]->d.sum_dims();
    auto&& d1 = xs[1]->d.sum_dims();
    if(d0 == d1) {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) && tbvec(*xs[1])).cast<float>();
    }
    else if(d0 < d1) {
      tbvec(fx).device(*dev.edevice) = (BVEC_BROADCAST_TO_DIM(tbvec(*xs[0]), xs[0]->d, fx.d) && tbvec(*xs[1])).cast<float>();
    }
    else {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) && BVEC_BROADCAST_TO_DIM(tbvec(*xs[1]), xs[1]->d, fx.d)).cast<float>();
    }
  }

  template<class MyDevice>
  void LogicalAnd::backward_dev_impl(const MyDevice& dev,
                                     const vector<const Tensor *>& xs,
                                     const Tensor& fx,
                                     const Tensor& dEdf,
                                     unsigned i,
                                     Tensor& dEdxi) const {
  }

  DYNET_NODE_INST_DEV_IMPL(LogicalAnd)


  // ************* LogicalOr *************
#ifndef __CUDACC__

  string LogicalOr::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "LogicalOr(" << arg_names[0] << "," << arg_names[1] << ')';
    return s.str();
  }

  Dim LogicalOr::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in LogicalOr")
    return xs[0].sum_dims() >= xs[1].sum_dims() ? xs[0] : xs[1];
  }

#endif

  template<class MyDevice>
  void LogicalOr::forward_dev_impl(const MyDevice& dev, const vector<const Tensor *>& xs, Tensor& fx) const {

    auto&& d0 = xs[0]->d.sum_dims();
    auto&& d1 = xs[1]->d.sum_dims();
    if(d0 == d1) {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) || tbvec(*xs[1])).cast<float>();
    }
    else if(d0 < d1) {
      tbvec(fx).device(*dev.edevice) = (BVEC_BROADCAST_TO_DIM(tbvec(*xs[0]), xs[0]->d, fx.d) || tbvec(*xs[1])).cast<float>();
    }
    else {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) || BVEC_BROADCAST_TO_DIM(tbvec(*xs[1]), xs[1]->d, fx.d)).cast<float>();
    }
  }

  template<class MyDevice>
  void LogicalOr::backward_dev_impl(const MyDevice& dev,
                                     const vector<const Tensor *>& xs,
                                     const Tensor& fx,
                                     const Tensor& dEdf,
                                     unsigned i,
                                     Tensor& dEdxi) const {
  }

  DYNET_NODE_INST_DEV_IMPL(LogicalOr)

  // ************* CwiseEq *************
#ifndef __CUDACC__

  string CwiseEq::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "CwiseEq(" << arg_names[0] << "," << arg_names[1] << ')';
    return s.str();
  }

  Dim CwiseEq::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in CwiseEq");
    return xs[0].sum_dims() >= xs[1].sum_dims() ? xs[0] : xs[1];
  }

#endif

  template<class MyDevice>
  void CwiseEq::forward_dev_impl(const MyDevice& dev, const vector<const Tensor *>& xs, Tensor& fx) const {
    auto&& d0 = xs[0]->d.sum_dims();
    auto&& d1 = xs[1]->d.sum_dims();

    if(d0 == d1) {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) == tbvec(*xs[1])).cast<float>();
    }
    else if(d0 < d1) {
      tbvec(fx).device(*dev.edevice) = (BVEC_BROADCAST_TO_DIM(tbvec(*xs[0]), xs[0]->d, fx.d) == tbvec(*xs[1])).cast<float>();
    }
    else {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) == BVEC_BROADCAST_TO_DIM(tbvec(*xs[1]), xs[1]->d, fx.d)).cast<float>();
    }
  }

  template<class MyDevice>
  void CwiseEq::backward_dev_impl(const MyDevice& dev,
                                   const vector<const Tensor *>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const {

  }

  DYNET_NODE_INST_DEV_IMPL(CwiseEq)

  // ************* CwiseNe *************
#ifndef __CUDACC__

  string CwiseNe::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "CwiseNe(" << arg_names[0] << "," << arg_names[1] << ')';
    return s.str();
  }

  Dim CwiseNe::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in CwiseNe");
    return xs[0].sum_dims() >= xs[1].sum_dims() ? xs[0] : xs[1];
  }

#endif

  template<class MyDevice>
  void CwiseNe::forward_dev_impl(const MyDevice& dev, const vector<const Tensor *>& xs, Tensor& fx) const {
    auto&& d0 = xs[0]->d.sum_dims();
    auto&& d1 = xs[1]->d.sum_dims();
    if(d0 == d1) {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) != tbvec(*xs[1])).cast<float>();
    }
    else if(d0 < d1) {
      tbvec(fx).device(*dev.edevice) = (BVEC_BROADCAST_TO_DIM(tbvec(*xs[0]), xs[0]->d, fx.d) != tbvec(*xs[1])).cast<float>();
    }
    else {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) != BVEC_BROADCAST_TO_DIM(tbvec(*xs[1]), xs[1]->d, fx.d)).cast<float>();
    }
  }

  template<class MyDevice>
  void CwiseNe::backward_dev_impl(const MyDevice& dev,
                                   const vector<const Tensor *>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const {

  }

  DYNET_NODE_INST_DEV_IMPL(CwiseNe)

  // ************* CwiseGt *************
#ifndef __CUDACC__

  string CwiseGt::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "CwiseGt(" << arg_names[0] << "," << arg_names[1] << ')';
    return s.str();
  }

  Dim CwiseGt::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in CwiseGt");
    return xs[0].sum_dims() >= xs[1].sum_dims() ? xs[0] : xs[1];
  }

#endif

  template<class MyDevice>
  void CwiseGt::forward_dev_impl(const MyDevice& dev, const vector<const Tensor *>& xs, Tensor& fx) const {
    auto&& d0 = xs[0]->d.sum_dims();
    auto&& d1 = xs[1]->d.sum_dims();
    if(d0 == d1) {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) > tbvec(*xs[1])).cast<float>();
    }
    else if(d0 < d1) {
      tbvec(fx).device(*dev.edevice) = (BVEC_BROADCAST_TO_DIM(tbvec(*xs[0]), xs[0]->d, fx.d) > tbvec(*xs[1])).cast<float>();
    }
    else {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) > BVEC_BROADCAST_TO_DIM(tbvec(*xs[1]), xs[1]->d, fx.d)).cast<float>();
    }
  }

  template<class MyDevice>
  void CwiseGt::backward_dev_impl(const MyDevice& dev,
                                   const vector<const Tensor *>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const {

  }

  DYNET_NODE_INST_DEV_IMPL(CwiseGt)

  // ************* CwiseLt *************
#ifndef __CUDACC__

  string CwiseLt::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "CwiseLt(" << arg_names[0] << "," << arg_names[1] << ')';
    return s.str();
  }

  Dim CwiseLt::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in CwiseLt");
    return xs[0].sum_dims() >= xs[1].sum_dims() ? xs[0] : xs[1];
  }

#endif

  template<class MyDevice>
  void CwiseLt::forward_dev_impl(const MyDevice& dev, const vector<const Tensor *>& xs, Tensor& fx) const {
    tvec(fx).device(*dev.edevice) = (tvec(*xs[0]) < tvec(*xs[1])).cast<float>();
    auto&& d0 = xs[0]->d.sum_dims();
    auto&& d1 = xs[1]->d.sum_dims();
    if(d0 == d1) {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) < tbvec(*xs[1])).cast<float>();
    }
    else if(d0 < d1) {
      tbvec(fx).device(*dev.edevice) = (BVEC_BROADCAST_TO_DIM(tbvec(*xs[0]), xs[0]->d, fx.d) < tbvec(*xs[1])).cast<float>();
    }
    else {
      tbvec(fx).device(*dev.edevice) = (tbvec(*xs[0]) < BVEC_BROADCAST_TO_DIM(tbvec(*xs[1]), xs[1]->d, fx.d)).cast<float>();
    }
  }

  template<class MyDevice>
  void CwiseLt::backward_dev_impl(const MyDevice& dev,
                                   const vector<const Tensor *>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const {

  }

  DYNET_NODE_INST_DEV_IMPL(CwiseLt)
}
