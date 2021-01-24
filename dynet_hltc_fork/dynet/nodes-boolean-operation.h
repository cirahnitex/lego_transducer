//
// Created by Dekai WU and YAN Yuchen on 20200630.
//

#ifndef DYNET_NODES_BOOLEAN_OPERATION_H
#define DYNET_NODES_BOOLEAN_OPERATION_H

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {
  struct ToBoolean : public Node {
    explicit ToBoolean(const std::initializer_list<VariableIndex>& a) : Node(a) {}
    virtual bool supports_multibatch() const override { return true; }
    DYNET_NODE_DEFINE_DEV_IMPL()
  };

  struct ConditionalExpression : public Node {
    explicit ConditionalExpression(const std::initializer_list<VariableIndex>& a) : Node(a) {}
    virtual bool supports_multibatch() const override { return true; }
    DYNET_NODE_DEFINE_DEV_IMPL()
  };

  struct LogicalNot : public Node {
    explicit LogicalNot(const std::initializer_list<VariableIndex>& a) : Node(a) {}
    virtual bool supports_multibatch() const override { return true; }
    DYNET_NODE_DEFINE_DEV_IMPL()
  };

  struct LogicalAnd : public Node {
    explicit LogicalAnd(const std::initializer_list<VariableIndex>& a) : Node(a) {}
    virtual bool supports_multibatch() const override { return true; }
    DYNET_NODE_DEFINE_DEV_IMPL()
  };

  struct LogicalOr : public Node {
    explicit LogicalOr(const std::initializer_list<VariableIndex>& a) : Node(a) {}
    virtual bool supports_multibatch() const override { return true; }
    DYNET_NODE_DEFINE_DEV_IMPL()
  };

  struct CwiseEq : public Node {
    explicit CwiseEq(const std::initializer_list<VariableIndex>& a) : Node(a) {}
    virtual bool supports_multibatch() const override { return true; }
    DYNET_NODE_DEFINE_DEV_IMPL()
  };

  struct CwiseNe : public Node {
    explicit CwiseNe(const std::initializer_list<VariableIndex>& a) : Node(a) {}
    virtual bool supports_multibatch() const override { return true; }
    DYNET_NODE_DEFINE_DEV_IMPL()
  };

  struct CwiseGt : public Node {
    explicit CwiseGt(const std::initializer_list<VariableIndex>& a) : Node(a) {}
    virtual bool supports_multibatch() const override { return true; }
    DYNET_NODE_DEFINE_DEV_IMPL()
  };

  struct CwiseLt : public Node {
    explicit CwiseLt(const std::initializer_list<VariableIndex>& a) : Node(a) {}
    virtual bool supports_multibatch() const override { return true; }
    DYNET_NODE_DEFINE_DEV_IMPL()
  };
}

#endif //DYNET_NODES_BOOLEAN_OPERATION_H
