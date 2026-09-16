#include "expr/api.h"
#include "nn/layer.h"
#include "optimiser/sgd.h"
#include "passes/compile.h"
#include "test_backend.h"
#include <cassert>
#include <cstddef>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

using Sample = std::pair<Mat, Mat>;
using SGD = Optimiser::SGD<Mat>;

std::vector<Sample> load_digits(std::string_view filename) {
  std::ifstream file(std::string{filename});
  std::vector<Sample> dataset;
  std::string line;

  while (std::getline(file, line)) {
    Mat input({1, 64});
    Mat target({1, 10});

    std::stringstream ss(line);
    std::string value;

    for (size_t i = 0; i < 64; i++) {
      std::getline(ss, value, ',');
      input.set(static_cast<kernel_type>(std::stod(value) / 16.0f), 0, i);
    }

    std::getline(ss, value, ',');
    const size_t label = static_cast<size_t>(std::stoi(value));
    target.set(1.0f, 0, label);

    dataset.emplace_back(std::move(input), std::move(target));
  }
  return dataset;
}

using namespace expr;
using namespace expr::tensor;
int main() {
  test_backend::init();
  test_backend::Runtime runtime;

  auto train_data =
      load_digits("/Users/matoli01/bob/graph/data/hand_write/optdigits.tra");
  auto test_data =
      load_digits("/Users/matoli01/bob/graph/data/hand_write/optdigits.tes");

  std::mt19937 gen(1);
  nn::Layer<Mat> layer1(64, 128, gen);
  nn::Layer<Mat> layer2(128, 64, gen);
  nn::Layer<Mat> layer3(64, 10, gen);

  Mat input_mat({1, 64});
  Mat root_grad({1, 10});
  for (size_t i = 0; i < 10; ++i) {
    root_grad.set(1.0f, 0, i);
  }
  Tensor input(std::move(input_mat));

  auto network = sigmoid(layer3(sigmoid(layer2(sigmoid(layer1(input))))));

  Tensor target(Mat({1, 10}));
  auto loss = expr::loss::sqe(network, target);

  auto optimiser_shared = SGD::generate_params(0.01f);
  auto compiled_loss =
      passes::compile<Mat, SGD>(loss, runtime, optimiser_shared);
  auto compiled_network =
      passes::compile<Mat, SGD>(network, runtime, optimiser_shared);
  compiled_loss.prepare();
  compiled_network.prepare();
  for (size_t epoch = 0; epoch < 10; ++epoch) {
    for (auto &[x, y] : train_data) {
      input.set_value(x);
      target.set_value(y);
      compiled_loss.zero_grad();
      (void)compiled_loss.forward();
      compiled_loss.backward(root_grad);
      compiled_loss.step();
    }
  }
  // tests data
  size_t correct = 0;
  for (auto &[x, y] : test_data) {
    input.set_value(x);
    Mat out = compiled_network.forward();
    size_t number = test_backend::argmax(out).get(0, 0);
    size_t excpected = test_backend::argmax(y).get(0, 0);
    if (excpected == number) {
      correct += 1;
    }
    // need to get maximum index
  }
  float accuracy =
      static_cast<float>(correct) / static_cast<float>(test_data.size());

  assert(accuracy > 0.7f);
  test_backend::shutdown();
}
