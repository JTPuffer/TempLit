#include <cassert>
#include <utility>
#include <vector>

// i want to make a type based graph compiler
// can we make it so there is a generic verdic taht can take operations and can
// tehn apply tehm using perfect forwarding later so i dont have to impliment
// operators for every type
// template <typename OP>
// concept operation = requires(const OP &value) {
//  { value() } -> std::same_as<Tensor>;
//};
// template <typename OP>
// concept Leaf = requires(const OP &value) {
//  { value.get() } -> std::same_as<int>;
//};

// phase 2
// need to make Tensors poiunt to a shared ptr of the actual Tensor object so
// tensor buff or somethiung so things are lightweight and can be copied etc
//  need to make Tensor type Compile time unique
//
//  need to test behavour
//  to 2 tests does matrix layer actually work  with a normal type ?? do i get
//  the correct result? does the forwards and backwards pass actually work.
//
//  phase 3
//  Ok we have some tests they all pass so technicaly we have everything we need
//  to build a more complex NN we can implement some basic NN now and then start
//  changing the backend to use cuda or som other thing
//
//  phase 4
//  what we need to do now is make a netowrk thing that owns its weigfhts and
//  its biases and its operations so we dont have to keep defiing them here for
//  more complex networks i want to be able to say Layer(input features, output
//  features) and then i want to be able to say Some sigmoid/Relu or something
//  function
//  what would be nice is some way to have layers defined as expressions so i
//  could say EXPR but then we need to worry about weights and bias ownership
//  Have an abstracton layer arround things that can return an expression which
//  we can use to link Up things So a goof abstraction is Netowrk class Network
//  can have Multiple layers and expressions etc
#include "expr/api.h"
#include "optimiser/sgd.h"
#include "passes/compile.h"
#include "test_backend.h"

using namespace expr;
using namespace expr::tensor;
using namespace test_backend;
using SGD = Optimiser::SGD<Mat>;

int main() {
  test_backend::init();
  Runtime runtime;
  std::vector<std::pair<Mat, Mat>> data = {
      {Mat::from_values({{0, 0}}), Mat::from_values({{0}})},
      {Mat::from_values({{0, 1}}), Mat::from_values({{1}})},
      {Mat::from_values({{1, 0}}), Mat::from_values({{1}})},
      {Mat::from_values({{1, 1}}), Mat::from_values({{0}})},
  };

  // input: 1x2
  Mat input_mat = Mat::from_values({
      {0, 0},
  });

  // hidden layer weights: 2x2
  Mat weight_1_mat = Mat::from_values({
      {0.5f, -0.3f},
      {0.8f, 0.2f},
  });

  // hidden bias: 1x2
  Mat bias_1_mat = Mat::from_values({
      {0.1f, -0.1f},
  });

  // second layer weights: 2x1
  Mat weight_2_mat = Mat::from_values({
      {0.7f},
      {-0.5f},
  });

  // output bias: 1x1
  Mat bias_2_mat = Mat::from_values({
      {0.2f},
  });

  Mat root_grad = Mat::from_values({
      {1},
  });

  Tensor input(std::move(input_mat));
  GradTensor weight_1(std::move(weight_1_mat));
  GradTensor bias_1(std::move(bias_1_mat));
  GradTensor weight_2(std::move(weight_2_mat));
  GradTensor bias_2(std::move(bias_2_mat));

  auto layer = sigmoid(
      add(mult(sigmoid(add(mult(input, weight_1), bias_1)), weight_2), bias_2));

  Tensor target(Mat::from_values({{0}}));
  auto loss = expr::loss::sqe(layer, target);

  auto optimiser_shared = SGD::generate_params(0.1f);
  auto compiled_loss =
      passes::compile<Mat, SGD>(loss, runtime, optimiser_shared);
  auto compiled_layer =
      passes::compile<Mat, SGD>(layer, runtime, optimiser_shared);
  compiled_loss.prepare();
  compiled_layer.prepare();
  for (size_t epoch = 0; epoch < 10000; ++epoch) {
    for (auto &[x, y] : data) {
      input.set_value(x);
      target.set_value(y);
      compiled_loss.zero_grad();
      (void)compiled_loss.forward();
      compiled_loss.backward(root_grad);
      compiled_loss.step();
    }
  }

  // check NN
  int correct = 0;
  for (auto &[x, y] : data) {
    input.set_value(x);
    if (argmax(y) == argmax(compiled_layer.forward())) {
      correct += 1;
    }
  }
  // can have 1 error
  assert(correct >= 3);
  test_backend::shutdown();
}
