#include "expr/cross-entropy.h"
#include "model_utils/Dataset.h"
#include "model_utils/archive.h"
#include "model_utils/bpe_tokeniser.h"
#include "model_utils/loader.h"
#include "my_metal/runtime.h"
#include "my_metal/storage.h"
#include "nn/GPT.h"
#include "optimiser/AdamW.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <passes/compile.h>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
constexpr uint32_t batch_size = 12;
constexpr uint32_t sequence_size = 192;
constexpr uint32_t vocab_size = 2048;
constexpr uint32_t embedding_size = 384;
constexpr uint32_t heads = 6;
constexpr size_t training_steps = 75000;
constexpr size_t loss_interval = 500;
constexpr size_t validation_batches = 30;

constexpr float peak_lr = 1e-4f;
constexpr float decay_lr = 1e-4f;
constexpr float final_lr = 1e-5f;

constexpr size_t warmup_steps = 2000;
constexpr size_t decay_start = 10000;
constexpr float pi = 3.14159265358979323846f;

using Mat = RTensor<metal::Metal, metal::kernel_type>;
using SGD = Optimiser::AdamW<Mat>;

template <typename CompiledLoss, typename Input, typename Targets>
double validate(CompiledLoss &compiled_loss, Loader &loader, Input &input,
                Targets &targets, std::vector<float> &loss_values) {
  double accumulated_loss = 0.0;

  for (size_t validation_batch = 0; validation_batch < validation_batches;
       ++validation_batch) {
    const auto &batch = loader.get_batch(batch_size, sequence_size);
    input.get_value().set(batch.inputs);
    targets.get_value().set(batch.targets);

    const Mat loss = compiled_loss.forward();
    loss.read(loss_values);
    for (const float value : loss_values) {
      accumulated_loss += value;
    }
  }

  return accumulated_loss /
         static_cast<double>(validation_batches * loss_values.size());
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: train <token-dataset> <tokeniser>\n";
    return 1;
  }

  metal::init();
  // loading daatset and stuff//////////////////////////////
  metal::Runtime runtime;
  std::ifstream dataset_input(argv[1], std::ios::binary);
  Dataset dataset = Dataset::load_tokens(dataset_input);
  Loader loader(dataset.training_split());
  Loader validation_loader(dataset.validation_split(), 2);

  std::ifstream tokeniser_input(argv[2]);
  auto tokeniser = BPETokenizer::load(tokeniser_input);
  if (tokeniser.vocabulary_size() != vocab_size) {
    throw std::runtime_error("tokeniser vocabulary does not match the model");
  }

  /////////////////////////////////////////////////////////

  // model instansiation ////////////////////////////////////////////////

  using IndexMat = typename Mat::template rebind<uint32_t>;

  IndexMat input_values({batch_size, sequence_size});
  IndexMat target_values({batch_size, sequence_size});

  expr::tensor::Tensor input(std::move(input_values));
  expr::tensor::Tensor targets(std::move(target_values));

  std::mt19937 gen(1);
  nn::GPT<Mat, vocab_size, sequence_size, embedding_size, heads> model(
      batch_size, sequence_size, gen);
  auto graph = model(input);

  auto loss = expr::loss::cross_entropy(graph, targets);

  auto compiled_loss =
      passes::compile<Mat, SGD>(loss, runtime, SGD::generate_params(peak_lr));

  compiled_loss.prepare();

  Mat root_grad({batch_size, sequence_size, 1});

  root_grad.set(std::vector<float>(
      batch_size * sequence_size,
      1.0f / static_cast<float>(batch_size * sequence_size)));
  /////////////////////////////////////////////////////////

  ////// training /////////////////////////////////////////
  std::vector<float> loss_values(batch_size * sequence_size);
  double accumulated_loss = 0.0;

  for (size_t training_step = 0; training_step < training_steps;
       ++training_step) {

    ///// update learning rate /////////////////
    float lr;
    if (training_step < warmup_steps) {
      lr = peak_lr * static_cast<float>(training_step + 1) /
           static_cast<float>(warmup_steps);
    } else if (training_step < decay_start) {
      lr = peak_lr;
    } else {
      const float progress =
          static_cast<float>(training_step - decay_start) /
          static_cast<float>(training_steps - decay_start - 1);

      lr = final_lr +
           0.5f * (decay_lr - final_lr) * (1.0f + std::cos(pi * progress));
    }
    compiled_loss.set_lr(lr);

    /////////////// actual training and backward pass /////////////

    const auto &batch = loader.get_batch(batch_size, sequence_size);
    input.get_value().set(batch.inputs);
    targets.get_value().set(batch.targets);

    compiled_loss.zero_grad();
    const Mat loss_value = compiled_loss.forward();
    loss_value.read(loss_values);

    // track loss for later //////////////////
    for (const float value : loss_values) {
      accumulated_loss += value;
    }

    // backwardpass and update ///////////////
    compiled_loss.backward(root_grad);
    compiled_loss.step(1.0f);

    ////////////////////// validation loss printing stuff ////////////////////
    if ((training_step + 1) % loss_interval == 0) {
      const double mean_training_loss =
          accumulated_loss /
          static_cast<double>(loss_interval * loss_values.size());
      const double mean_validation_loss = validate(
          compiled_loss, validation_loader, input, targets, loss_values);
      std::cout << "step " << training_step + 1 << ": training loss "
                << mean_training_loss << ", validation loss "
                << mean_validation_loss << "loss:" << lr << '\n';
      accumulated_loss = 0.0;
    }

    if ((training_step + 1) % 1000 == 0) {
      const auto step = training_step + 1;
      const auto checkpoint =
          std::filesystem::path{"checkpoints"} /
          ("tiny_stories_step_" + std::to_string(step) + ".bin");

      std::filesystem::create_directories(checkpoint.parent_path());

      Archive archive;
      compiled_loss.save(archive);
      compiled_loss.save_optimisers(archive);

      std::ofstream output(checkpoint, std::ios::binary | std::ios::trunc);
      archive.commit(output);
    }
  }

  metal::shutdown();
}
