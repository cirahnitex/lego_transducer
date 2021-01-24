Lego library documentation {#mainpage}
========

Lego library is a deep learning library written in C++, that enables users to easily define and train **dynamic neural networks**.

### Motivation

#### Building dynamic neural network is painful

Building a neural network that changes topology depending on each training example is very clunky in mainstream deep learning frameworks like Tensorflow or Pytorch. For example, if you want to build a Tree-LSTM, usually you need to preprocess the tree structure into unintuitive adjacency tensors (as described in [this github repo](https://github.com/unbounce/pytorch-tree-lstm)) and then carefully engineer some tensor operations over the adjacency tensors. However, such approach is error-prone and not scalable. What if you want to skip some children depending on some conditions? What if you want to use one LSTM cell for Noun phrases and another LSTM cell for Verb phrases? It requires enormous linear algebra engineering to implement even a simple idea.

#### Our solution

Lego library solves the problem of defining dynamic neural networks by providing a collection of basic building blocks (like dense layers, arithmetic functions) and let the user to combine the building blocks in an extremely flexible way. It is so flexible, that you can even build a model out of basic comparison operations, that can perform quick sort on its input.

### Installation

[How to build Lego library](tutorials/installation.md)

### Getting started

[Lego library basics](tutorials/basics.md)
[Example: Building an XOR model](tutorials/examples/xor/index.md)
[Example: Building a naive RNN POS tagger model](tutorials/examples/rnn_diy/index.md)
