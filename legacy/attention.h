{
  metal::init();
  using Mat = RTensor<metal::Metal, metal::kernel_type>;
  metal::Runtime runtime;

  // Scaled dot-product self-attention:
  //
  // Q = X * Wq
  // K = X * Wk
  // V = X * Wv
  //
  // scores = (Q * transpose(K)) / sqrt(d_k)
  //
  // attention_weights = softmax(scores)
  //
  // output = attention_weights * V
  //
  // Combined:
  //
  // Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
  //
  // For self-attention:
  //
  // Attention(X) = softmax(
  //     ((X * Wq) * transpose(X * Wk)) / sqrt(d_k)
  // ) * (X * Wv)
  //
  // Shapes:
  //
  // X       : [seq_len, d_model]
  // Wq      : [d_model, d_k]
  // Wk      : [d_model, d_k]
  // Wv      : [d_model, d_v]
  //
  // Q       : [seq_len, d_k]
  // K       : [seq_len, d_k]:w
  // V       : [seq_len, d_v]
  //
  // Q * K^T : [seq_len, seq_len]
  // output  : [seq_len, d_v]
  uint32_t d_k = 128;
  Layer<Mat> query(64, d_k);
  Layer<Mat> key(64, d_k);
  Layer<Mat> value(64, 64);

  Mat input_mat({4, 64});
  expr::tensor::Tensor input(std::move(input_mat));

  expr::tensor::Tensor d_k_tensor(Mat(static_cast<metal::kernel_type>(d_k)));

  auto scores = expr::div(expr::mult(query(input), expr::transpose(key(input))),
                          expr::sqrt(d_k_tensor));

  auto weights = expr::soft_max(scores);

  auto attention = expr::mult(weights, value(input));
  expr::tensor::Tensor target(Mat({4, 64}));
  auto loss = expr::loss::sqe(attention, target);

  Mat root_grad({4, 64});
  for (size_t i = 0; i < root_grad.shape()[0]; ++i) {
    for (size_t j = 0; j < root_grad.shape()[1]; ++j) {
      root_grad.set(1.0f, i, j);
    }
  }

  auto compiled_loss = passes::compile<Mat>(loss, runtime);
  auto compiled_weights = passes::compile<Mat>(weights, runtime);
  auto compiled_attention = passes::compile<Mat>(attention, runtime);
  compiled_loss.prepare();
  compiled_weights.prepare();
  compiled_attention.prepare();
  Optimiser::SGD<Mat> sgd(0.1f);

  for (size_t sample = 0; sample < 10000; ++sample) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(sample));
    std::uniform_int_distribution<size_t> dist(0, 63);

    Mat temp_mat({4, 64});
    temp_mat.set(1.0f, 0, dist(rng));
    temp_mat.set(1.0f, 1, dist(rng));
    temp_mat.set(1.0f, 2, dist(rng));
    temp_mat.set(1.0f, 3, dist(rng));

    input.set_value(temp_mat);
    target.set_value(temp_mat);
    compiled_loss.zero_grad();
    (void)compiled_loss.forward();
    compiled_loss.backward(root_grad);
    compiled_loss.step(sgd);
  }
  std::mt19937 rng(9999);
  std::uniform_int_distribution<size_t> dist(0, 63);

  Mat test_mat({4, 64});

  size_t a = dist(rng);
  size_t b = dist(rng);
  size_t c = dist(rng);
  size_t d = dist(rng);

  test_mat.set(1.0f, 0, a);
  test_mat.set(1.0f, 1, b);
  test_mat.set(1.0f, 2, c);
  test_mat.set(1.0f, 3, d);
  input.set_value(test_mat);

  auto output = compiled_weights.forward();

  std::cout << output << '\n';

  for (size_t sample = 0; sample < 2; ++sample) {
    std::mt19937 sample_rng(static_cast<std::mt19937::result_type>(sample));
    std::uniform_int_distribution<size_t> sample_dist(0, 63);

    Mat temp_mat({4, 64});
    temp_mat.set(1.0f, 0, sample_dist(sample_rng));
    temp_mat.set(1.0f, 1, sample_dist(sample_rng));
    temp_mat.set(1.0f, 2, sample_dist(sample_rng));
    temp_mat.set(1.0f, 3, sample_dist(sample_rng));

    input.set_value(temp_mat);
    auto attention_output = compiled_attention.forward();
    std::cout << temp_mat << '\n';
    std::cout << attention_output << '\n';
  }
  metal::shutdown();
}
