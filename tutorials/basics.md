Lego library basics
========

In machine learning, you build a model that takes some input features and returns some predictions. Lego library is designed with exactly the same philosophy. In Lego library, you build a **transducer** that takes one or more values and returns a value after which the Lego library takes care of training it. Lego library provides many built-in transducers as primitive building blocks, such as dense layers, arithmetic functions, list manipulation functions etc. Lego library also allows you to compose transducers together to build complex transducers.

A transducer is a pure function (a function without side effect) and composing transducer models are analogous to composing functions. Mathematically, building a transducer model is Turing complete, but framed in a functional programming style as oppose to a procedural programming style.

A transducer is more than just a pure function in the sense that a transducer may have internal parameters that can be trained by performing backpropagation. In summary, Lego library allows you to build arbitrarily complex transducer models, after which can easily be trained, saved and loaded.

### Using a transducer
A **transducer** (`tg::transducer_model`) takes one or more **value**s (`tg::value_t`) and returns a **value**. For example, the following code snippet applies tanh transducer on a tensor value:
```cpp
// define the input tensor
value_t input_tensor(tensor_t({-1,1,2}));

// apply tanh operation on the input tensor
value_t output_tensor = tg::tanh(input_tensor);

// print the output tensor
cout << output_tensor << endl;
```

### The value type
A value (`tg::value_t`) can hold one of the followings:
  * null
  * integer: an integer, usually representing an index
  * float: a single precision floating point number
  * tensor: a float-valued tensor of any rank
  * boolean: an alias to integer type, where 0 means false and other value means true.
  * symbol: a string
  * list: an ordered list of values (may contain nested lists)
  
Here are examples of how these values can be constructed:
```cpp
value_t my_null(nullptr);
value_t my_integer(7);
value_t my_float(-0.8);
value_t my_tensor_2_by_3(tensor_t({1.0,2.0,3.0,4.0,5.0,6.0}, {2,3}));
value_t my_boolean(false);
value_t my_symbol("cat");
value_t my_list_of_3_items = make_list(my_null, my_integer, my_float);
```

Usually, a transducer may not accepts all input value types. For example, `tg::tanh` transducer only takes integer, float or tensor.

### Using built-in transducer factories

Besides built-in transducers like `tg::tanh`, a transducer can be created from built-in transducer factories. For example, here is how you can use the `tg::dense_structure` transducer factory to create a dense layer.

```cpp
// Create a randomly initialized dense layer that
// * takes a tensor of shape {4}
// * returns a tensor of shape {2}
transducer_model dense = dense_structure.initialize(4, 2);

// Create some arbitrary input tensor that will be used to test the dense layer
value_t input(tensor_t({1, -1, 0.5, 0.9}));

// Apply the dense layer on the input
value_t output = dense(input);

// Print the output tensor
cout << output << endl;
```

### Building complex transducers using `compose` syntax

A tanh dense layer can be think of a tanh operation composed with a dense layer. Formally, a composition of function y=f(x) and function y=g(x) is y=f(g(x)). Here is an example code of building a tanh dense layer:
```cpp
// Create a dense layer
transducer_model linear_dense = dense_structure.initialize(4,2);

// Create a tanh dense layer by composing a tanh operation with a dense layer
transducer_model tanh_dense = compose(tg::tanh, linear_dense);

// Create some arbitrary input tensor that will be used to test the tanh dense layer
value_t input(tensor_t({1, -1, 0.5, 0.9}));

// Print the output tensor
cout << tanh_dense(input) << endl;
```

### Building complex transducers using lambda syntax

The compose syntax works well for unary transducers, but gets clunky when dealing with binary transducers. Consider the following example. We have the following transducers:
  * a unary transducer y=f1(x)
  * another unary transducer y=f2(x)
  * a binary transducer y=g(x1, x2)
And we want to build the transducer y=g(f1(x1), f2(x2)). As an be seen, there is no clear way of how to do this using the compose syntax.

So, instead of doing it in compose syntax, we could do it in lambda syntax in the following way:
```cpp
// Use an arbitrary unary function as an example of f1 and f2
transducer_model f1 = dense_structure.initialize(4,2);
transducer_model f2 = dense_structure.initialize(4,2);

// Use another arbitrary binary function as an example of g
transducer_model g = biaffine_structure.initialize(2,2,4);

// Build the transducer y=g(f1(x1),f2(x2)) using lambda syntax
transducer_model h = transducer_model([&](const value_t& x1, const valule_t& x2) {
  value_t y1 = f1(x1);
  value_t y2 = f2(x2);
  return g(y1, y2);
});
```

As we can see, our desired transducer can be constructed using the lambda syntax easily.

#### Drawbacks of lambda syntax and how to fix it

However, constructing a transducer from a user-supplied C++ lambda function can be challenging implemtational-wise. One naive implementation is to store the C++ lamdba function in the transducer, so that later when applying the transducer on concrete values, the C++ lambda function is invoked to compute the output. However, this implementation has severe drawbacks:
  * The transducer only holds the C++ lamdba function as a black box without any information about the internal topology, which makes automatic computation of derivatives using chain rule impossible.
  * The execution of the C++ lambda function cannot be optimized. In the above example, the computation of f1 and f2 could potentially be parallelized but cannot be done.
  * The C++ lambda function cannot be serialized into a file which means this transducer cannot be saved/loaded.

The solution of all these problems is that the composed transducer should store the internal transducer network topology instead of a native C++ lamdba function. This could be achieved by tracking the execution of the C++ lambda function. Formally, the user-supplied C++ lambda function no longer operates on concrete values at transduction time. Instead, it operates on **value placeholders** at model construction time. A value placeholder is something provided by our library, that instead of holding concrete values, records the transducer network topology. With this new syntax, the lambda syntax will look like this:

```cpp
// Use an arbitrary unary function as an example of f1 and f2
transducer_model f1 = dense_structure.initialize(4,2);
transducer_model f2 = dense_structure.initialize(4,2);

// Use another arbitrary binary function as an example of g
transducer_model g = biaffine_structure.initialize(2,2,4);

// Build the transducer y=g(f1(x1),f2(x2)) using value placeholder lambda syntax
transducer_model h = transducer_model([&](const value_placeholder& x1, const value_placeholder& x2) {
  value_placeholder y1 = f1(x1);
  value_placeholder y2 = f2(x2);
  return g(y1, y2);
});
```

#### Nested lambda transducers

A lambda transducer can be declared nested with another lambda transducer, like this:

```cpp
transducer_model my_distance_2d = transducer_model([&](
  const value_placeholder& x1, const value_placeholder& y1,
  const value_placeholder& x2, const value_placeholder& y2)->value_placeholder{

  transducer_model my_square_distance_1d = transducer_model([&](
    const value_placeholder& x1, const value_placeholder& x2)->value_placeholder {
    value_placeholder d = x1 - x2;
    return d * d;
  });

  return tg::sqrt(my_square_distance_1d(x1, x2) + my_square_distance_1d(y1, y2));
});

cout << my_distance_2d(1,1,-2,-2) << endl; // 4.24264
```

#### Value capturing in nested lambda transducers

A typical usage for such pattern is for the inner transducer to use the value placeholders declared in the outer transducer, similar to the **capture** concept in C++. For example, here is a transducer that computes f(x,y,z)=(x+z)*(y+z);

```cpp
transducer_model my_transducer = transducer_model([&](
  const value_placeholder& x, const value_placeholder& y, const value_placeholder& z)
  ->value_placeholder{

  transducer_model add_z = transducer_model([&](
    const value_placeholder& t)->value_placeholder {
    return t + z; // The value z is captured from the outside
  });

  return add_z(x) * add_z(y);
});

cout << my_transducer(3, 2, 1) << endl; // 12
```

#### Recursion

A lambda transducer can recursively invoke itself. For example, here is how factorial can be defined using lambda transducer syntax:

```cpp
transducer_model my_factorial; // Pre-declare this transducer so that it can be referred to within its own definition.
my_factorial = transducer_model([&](const value_placeholder& x)->value_placeholder{
  value_placeholder zero = value_placeholder::constant(0);
  value_placeholder one = value_placeholder::constant(1);
  return lazy_ifelse(x == zero, one, x * my_factorial(x-1));
});

cout << my_factorial(5) << endl; // 120
```

This example uses the `tg::lazy_ifelse` transducer to perform conditional branching. `tg::lazy_ifelse` returns the second argument when the first argument is true, and returns the third argument otherwise, similar to how the ternary operator in C++ works.

### Saving and loading

A transducer can be easily saved and loaded:
```cpp
transducer_model my_factorial;
my_factorial = transducer_model([&](const value_placeholder& x)->value_placeholder{
  value_placeholder zero = value_placeholder::constant(0);
  value_placeholder one = value_placeholder::constant(1);
  return lazy_ifelse(x == zero, one, x * my_factorial(x-1));
});

// Saving the transducer
transducer_model::save_to_file("model.bin", my_factorial);

// Loading the transducer
transducer_model loaded_model;
transducer_model::load_from_file("model.bin", loaded_model);

cout << loaded_model(5) << endl; // 120
```


### Debugging value placeholders

When compared with the naive implementation, the value placeholder implementation is more difficult for the user to debug. This is because, the concrete value of a value placeholder is unavailable at model construction time, which means the value cannot be printed. Consider the following example:

```cpp
// Create a randomly initialized dense layer
transducer_model linear_dense = dense_structure.initialize(4,2);

// Create a tanh dense layer using the lambda syntax
transducer_model tanh_dense([&](const value_placeholder& x)->value_placeholder {
  value_placeholder y = linear_dense(x);

  // trying to see the output value of linear_dense, GIVES A COMPILATION ERROR!
  cout << "Value after linear_dense: " << y << endl;

  return tanh(y);
});

// Create some arbitrary input tensor that will be used to test the tanh dense layer
value_t input(tensor_t({1, -1, 0.5, 0.9}));

// Apply the tanh dense layer
value_t output = tanh_dense(input);

// Print the output tensor
cout << tanh_dense(input) << endl;
```

Compiling the above example gives an error, because a value placeholder cannot be printed. This is because, by the time that value placeholder `y` is constructed, it only holds the network topology instead of holding concrete data. To solve this problem, our library provides a transducer called **trace**, that lets its input passes through but have the side effect of printing it. Here is how you can use trace to achieve what was intended in the previous code:

```cpp
// Create a randomly initialized dense layer
transducer_model linear_dense = dense_structure.initialize(4,2);

// Create a tanh dense layer using the lambda syntax
transducer_model tanh_dense([&](const value_placeholder& x)->value_placeholder {
  value_placeholder y = linear_dense(x);

  // A trace transducer lets its input passes through while having the side effect of printing it,
  // with an optional prefix.
  y = trace(y, "Value after linear_dense: ");

  return tg::tanh(y);
});

// Create some arbitrary input tensor that will be used to test the tanh dense layer
value_t input(tensor_t({1, -1, 0.5, 0.9}));

// Apply the tanh dense layer
value_t output = tanh_dense(input); 

// Print the output tensor
cout << tanh_dense(input) << endl;
```
