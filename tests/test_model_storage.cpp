#include "CPU/runtime.h"
#include "expr/api.h"
#include "model_utils/archive.h"
#include "optimiser/AdamW.h"
#include "optimiser/sgd.h"
#include "passes/compile.h"

#include <array>
#include <cstdint>
#include <sstream>
#include <span>
#include <stdexcept>

static_assert(ArchiveType<Archive>);
static_assert(ArchiveType<ArchiveView>);

using Mat = RTensor<cpu::CPU, cpu::kernel_type>;
using AdamW = Optimiser::AdamW<Mat>;
using SGD = Optimiser::SGD<Mat>;

void test_adamw_state_round_trip() {
  const Mat parameter = Mat::from_values({{1.0f, 2.0f}});
  AdamW optimiser;
  optimiser.initialise_for(parameter);

  Archive archive;
  auto state = archive.scope("optimiser");
  optimiser.save(state);

  AdamW restored;
  restored.initialise_for(parameter);
  restored.load(state);

  Archive restored_archive;
  auto restored_state = restored_archive.scope("optimiser");
  restored.save(restored_state);
}

bool test_compiled_save_and_load(Archive &archive) {
  cpu::Runtime runtime;

  expr::tensor::Tensor input(Mat::from_values({{2.0f, 3.0f}}));
  expr::tensor::GradTensor parameter(
      Mat::from_values({{4.0f, 5.0f}}));
  auto expression = expr::hadamard(input, parameter);

  auto compiled = passes::compile<Mat, SGD>(
      expression, runtime, SGD::generate_params(0.0f));
  compiled.prepare();

  const Mat expected = Mat::from_values({{8.0f, 15.0f}});
  const bool initial_output_is_correct = compiled.forward() == expected;

  compiled.save(archive);
  parameter.set_value(Mat::from_values({{1.0f, 1.0f}}));
  const bool parameter_was_replaced =
      compiled.forward() == Mat::from_values({{2.0f, 3.0f}});

  compiled.load(archive);
  const bool parameter_was_restored = compiled.forward() == expected;

  return initial_output_is_correct && parameter_was_replaced &&
         parameter_was_restored;
}

int main() {
  test_adamw_state_round_trip();

  const std::array<float, 4> expected_values = {1.5f, -2.0f, 3.25f, 4.0f};
  const std::array<uint32_t, 3> expected_indices = {7, 42, 1024};

  Archive archive;
  archive.write("values", std::as_bytes(std::span(expected_values)));
  auto indices_archive = archive.scope("indices");
  indices_archive.write("values",
                        std::as_bytes(std::span(expected_indices)));

  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  archive.commit(stream);
  stream.seekg(0);

  Archive restored;
  restored.restore(stream);

  std::array<float, 4> values{};
  std::array<uint32_t, 3> indices{};
  restored.read("values", std::as_writable_bytes(std::span(values)));
  auto restored_indices = restored.scope("indices");
  restored_indices.read("values",
                        std::as_writable_bytes(std::span(indices)));

  const bool direct_round_trip =
      values == expected_values && indices == expected_indices;
  Archive compiled_archive;
  const bool compiled_round_trip =
      test_compiled_save_and_load(compiled_archive);

  if (!direct_round_trip && !compiled_round_trip) {
    throw std::runtime_error(
        "archive and compiled model storage round trips failed");
  }
  if (!direct_round_trip) {
    throw std::runtime_error("archive round trip failed");
  }
  if (!compiled_round_trip) {
    throw std::runtime_error("compiled model save/load round trip failed");
  }
}
