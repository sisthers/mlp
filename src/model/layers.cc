#include "layers.h"

namespace s21 {
Layers::Layers(size_t hidden_layers_count)
    : hidden_layers_count_(hidden_layers_count) {
  deltas_for_weights_.reserve(hidden_layers_count + 1);
  deltas_for_biases_.reserve(hidden_layers_count + 1);

  deltas_for_weights_.push_back(
      S21Matrix<double>(kNeuronsOnHiddenLayerCount, kInputNeuronsCount));
  for (size_t layer = 0; layer < hidden_layers_count; ++layer) {
    if (layer != hidden_layers_count - 1)
      deltas_for_weights_.push_back(
          S21Matrix<double>(kNeuronsOnHiddenLayerCount));
    deltas_for_biases_.push_back(
        S21Matrix<double>(kNeuronsOnHiddenLayerCount, 1));
  }
  deltas_for_weights_.push_back(
      S21Matrix<double>(kOutputNeuronsCount, kNeuronsOnHiddenLayerCount));
  deltas_for_biases_.push_back(S21Matrix<double>(kOutputNeuronsCount, 1));
}

Layers::~Layers() {}

void Layers::UpdateWeights(std::vector<S21Matrix<double>>& weights,
                           std::vector<S21Matrix<double>>& biases,
                           double learning_rate) {
  for (size_t layer = 0; layer < hidden_layers_count_ + 1; ++layer) {
    biases[layer] -=
        learning_rate / mini_batch_size_ * deltas_for_biases_[layer];
    weights[layer] -=
        learning_rate / mini_batch_size_ * deltas_for_weights_[layer];
  }
}

void Layers::ResetDeltas() {
  for (size_t layer = 0; layer < hidden_layers_count_ + 1; ++layer) {
    deltas_for_biases_[layer] *= 0;
    deltas_for_weights_[layer] *= 0;
  }
}

void Layers::SetMiniBatchSize(size_t size) {
  if (size == 0) throw std::runtime_error("Invalid size");
  mini_batch_size_ = size;
}

size_t Layers::GetMiniBatchSize() const noexcept { return mini_batch_size_; }

MatrixLayers::MatrixLayers(size_t hidden_layers_count)
    : Layers(hidden_layers_count) {
  neurons_.reserve(hidden_layers_count + 2);
  neurons_.push_back(S21Matrix<double>(kInputNeuronsCount, 1));
  for (size_t layer = 0; layer < hidden_layers_count; ++layer) {
    neurons_.push_back(S21Matrix<double>(kNeuronsOnHiddenLayerCount, 1));
  }
  neurons_.push_back(S21Matrix<double>(kOutputNeuronsCount, 1));
}

MatrixLayers::~MatrixLayers() {}

void MatrixLayers::FeedForward(const S21Matrix<double>& image,
                               const std::vector<S21Matrix<double>>& weights,
                               const std::vector<S21Matrix<double>>& biases) {
  neurons_.front() = image;
  for (size_t layer = 1; layer < hidden_layers_count_ + 2; ++layer) {
    neurons_[layer] =
        weights[layer - 1] * neurons_[layer - 1] + biases[layer - 1];
    neurons_[layer].UseFunction(Sigmoid::SigmoidFunction);
  }
}

size_t MatrixLayers::GetMaxOutputIndex() const noexcept {
  double max = neurons_.back()(0, 0);
  size_t max_index = 0;
  for (size_t row = 1; row < kOutputNeuronsCount; ++row) {
    double value = neurons_.back()(row, 0);
    if (value > max) {
      max = value;
      max_index = row;
    }
  }
  return max_index;
}

void MatrixLayers::ChangeNumberOfHiddenLayers(size_t number) {
  if (number < 2 || number > 5)
    throw std::runtime_error(
        "Number of hidden layers should be between 2 and 5");
  if (number == hidden_layers_count_) return;
  if (number > hidden_layers_count_) {
    neurons_.insert(neurons_.end() - 1, number - hidden_layers_count_,
                    S21Matrix<double>(kNeuronsOnHiddenLayerCount, 1));
    deltas_for_weights_.insert(deltas_for_weights_.end() - 1,
                               number - hidden_layers_count_,
                               S21Matrix<double>(kNeuronsOnHiddenLayerCount));

    deltas_for_biases_.insert(deltas_for_biases_.end() - 1,
                              number - hidden_layers_count_,
                              S21Matrix<double>(kNeuronsOnHiddenLayerCount, 1));
  } else {
    neurons_.erase(neurons_.begin() + 1,
                   neurons_.begin() + 1 + hidden_layers_count_ - number);
    deltas_for_weights_.erase(
        deltas_for_weights_.begin() + 1,
        deltas_for_weights_.begin() + 1 + hidden_layers_count_ - number);
    deltas_for_biases_.erase(
        deltas_for_biases_.begin(),
        deltas_for_biases_.begin() + hidden_layers_count_ - number);
  }
  hidden_layers_count_ = number;
}

void MatrixLayers::BackPropogation(unsigned char expected_result,
                                   std::vector<S21Matrix<double>>& weights) {
  auto deltas = deltas_for_biases_;

  for (size_t row = 0; row < kOutputNeuronsCount; ++row) {
    auto neuron_value = neurons_.back()(row, 0);
    double exp_res = 0.0;
    if (row == static_cast<size_t>(expected_result - 97)) exp_res = 1.0;
    deltas.back()(row, 0) =
        Sigmoid::SigmoidDerivative(neuron_value) * (neuron_value - exp_res);
  }

  for (size_t layer = hidden_layers_count_; layer > 0; --layer) {
    deltas[layer - 1] = weights[layer].Transpose() * deltas[layer];
    for (size_t row = 0; row < deltas[layer - 1].GetRows(); ++row) {
      auto neuron_value = neurons_[layer](row, 0);
      deltas[layer - 1](row, 0) *= Sigmoid::SigmoidDerivative(neuron_value);
    }
  }

  for (size_t layer = 0; layer < hidden_layers_count_ + 1; ++layer) {
    deltas_for_biases_[layer] += deltas[layer];
    deltas_for_weights_[layer] += deltas[layer] * neurons_[layer].Transpose();
  }
}

double MatrixLayers::TotalCost(unsigned char expected_result) const {
  double sum = 0;
  for (size_t row = 0; row < kOutputNeuronsCount; ++row) {
    double value = neurons_.back()(row, 0);
    if (row == static_cast<size_t>(expected_result - 97)) {
      sum += (1.0 - value) * (1.0 - value);
    } else {
      sum += (0.0 - value) * (0.0 - value);
    }
  }
  return sum / 2.0;
}

GraphLayers::GraphLayers(size_t hidden_layers_count)
    : Layers(hidden_layers_count) {
  root_0_0_neuron_ = new Neuron();
  root_0_0_neuron_->outputs.reserve(kNeuronsOnHiddenLayerCount);
  for (size_t i = 0; i < kNeuronsOnHiddenLayerCount; ++i)
    root_0_0_neuron_->outputs.push_back(new Neuron());
  root_0_0_neuron_->outputs.front()->inputs.reserve(kInputNeuronsCount);
  root_0_0_neuron_->outputs.front()->inputs.push_back(root_0_0_neuron_);
  for (size_t i = 1; i < kInputNeuronsCount; ++i) {
    root_0_0_neuron_->outputs.front()->inputs.push_back(new Neuron());
    root_0_0_neuron_->outputs.front()->inputs[i]->outputs =
        root_0_0_neuron_->outputs;
  }
  auto this_layer = root_0_0_neuron_;
  auto next_layer = root_0_0_neuron_->outputs.front();
  for (size_t i = 0; i < hidden_layers_count_; ++i) {
    if (i != hidden_layers_count_ - 1)
      next_layer->outputs.reserve(kNeuronsOnHiddenLayerCount);
    else
      next_layer->outputs.reserve(kOutputNeuronsCount);
    for (size_t j = 0; j < next_layer->outputs.capacity(); ++j)
      next_layer->outputs.push_back(new Neuron());
    for (size_t j = 1; j < kNeuronsOnHiddenLayerCount; ++j) {
      this_layer->outputs[j]->inputs = next_layer->inputs;
      this_layer->outputs[j]->outputs = next_layer->outputs;
    }
    this_layer = next_layer;
    next_layer = this_layer->outputs.front();
    next_layer->inputs = this_layer->inputs.front()->outputs;
  }
  for (size_t i = 1; i < kOutputNeuronsCount; ++i) {
    this_layer->outputs[i]->inputs = next_layer->inputs;
  }
  pre_last_layer_neuron_ = this_layer;
}

GraphLayers::~GraphLayers() {
  for (size_t i = 0; i < hidden_layers_count_; ++i) {
    for (size_t j = 0; j < pre_last_layer_neuron_->outputs.size(); ++j) {
      delete pre_last_layer_neuron_->outputs[j];
    }
    if (i != hidden_layers_count_ - 1)
      pre_last_layer_neuron_ = pre_last_layer_neuron_->inputs.front();
  }
  auto last_inputs = pre_last_layer_neuron_->inputs;
  pre_last_layer_neuron_ = pre_last_layer_neuron_->inputs.front();

  for (size_t j = 0; j < pre_last_layer_neuron_->outputs.size(); ++j)
    delete pre_last_layer_neuron_->outputs[j];
  for (size_t j = 0; j < last_inputs.size(); ++j) delete last_inputs[j];
}

void GraphLayers::FeedForward(const S21Matrix<double>& image,
                              const std::vector<S21Matrix<double>>& weights,
                              const std::vector<S21Matrix<double>>& biases) {
  auto inputs = root_0_0_neuron_->outputs.front()->inputs;
  for (size_t i = 0; i < kInputNeuronsCount; ++i) inputs[i]->val = image(i, 0);

  auto layer = root_0_0_neuron_;
  for (size_t i = 0; i < hidden_layers_count_ + 1; ++i) {
    for (size_t neuron_idex = 0; neuron_idex < layer->outputs.size();
         ++neuron_idex) {
      double sum = 0;
      for (size_t w_idex = 0; w_idex < layer->outputs.front()->inputs.size();
           ++w_idex) {
        sum += weights[i](neuron_idex, w_idex) *
               layer->outputs.front()->inputs[w_idex]->val;
      }
      sum += biases[i](neuron_idex, 0);
      sum = Sigmoid::SigmoidFunction(sum);
      layer->outputs[neuron_idex]->val = sum;
    }
    layer = layer->outputs.front();
  }
}

size_t GraphLayers::GetMaxOutputIndex() const noexcept {
  double max = pre_last_layer_neuron_->outputs.front()->val;
  size_t max_index = 0;
  for (size_t row = 1; row < kOutputNeuronsCount; ++row) {
    double value = pre_last_layer_neuron_->outputs[row]->val;
    if (value > max) {
      max = value;
      max_index = row;
    }
  }
  return max_index;
}

void GraphLayers::ChangeNumberOfHiddenLayers(size_t number) {
  if (number < 2 || number > 5)
    throw std::runtime_error(
        "Number of hidden layers should be between 2 and 5");
  if (number == hidden_layers_count_) return;
  auto ins = root_0_0_neuron_->outputs.front()->inputs;
  auto outs = root_0_0_neuron_->outputs;
  ins.front()->outputs.clear();
  if (number > hidden_layers_count_) {
    deltas_for_weights_.insert(deltas_for_weights_.end() - 1,
                               number - hidden_layers_count_,
                               S21Matrix<double>(kNeuronsOnHiddenLayerCount));

    deltas_for_biases_.insert(deltas_for_biases_.end() - 1,
                              number - hidden_layers_count_,
                              S21Matrix<double>(kNeuronsOnHiddenLayerCount, 1));

    outs.front()->inputs.clear();
    while (hidden_layers_count_ < number) {
      for (size_t i = 0; i < kNeuronsOnHiddenLayerCount; ++i)
        ins.front()->outputs.push_back(new Neuron());
      for (size_t i = 1; i < ins.size(); ++i)
        ins[i]->outputs = ins.front()->outputs;
      for (size_t i = 0; i < ins.front()->outputs.size(); ++i)
        ins.front()->outputs[i]->inputs = ins;
      ins = ins.front()->outputs;
      ++hidden_layers_count_;
    }
    for (size_t i = 0; i < kNeuronsOnHiddenLayerCount; ++i) {
      ins[i]->outputs = outs;
      outs[i]->inputs = ins;
    }
  } else {
    deltas_for_weights_.erase(
        deltas_for_weights_.begin() + 1,
        deltas_for_weights_.begin() + 1 + hidden_layers_count_ - number);
    deltas_for_biases_.erase(
        deltas_for_biases_.begin(),
        deltas_for_biases_.begin() + hidden_layers_count_ - number);

    while (number < hidden_layers_count_) {
      auto next = outs.front()->outputs;
      for (size_t i = 0; i < kNeuronsOnHiddenLayerCount; ++i) delete outs[i];
      outs = next;
      for (size_t i = 0; i < kNeuronsOnHiddenLayerCount; ++i) {
        ins[i]->outputs = outs;
        outs[i]->inputs = ins;
      }
      --hidden_layers_count_;
    }
  }
}

void GraphLayers::BackPropogation(unsigned char expected_result,
                                  std::vector<S21Matrix<double>>& weights) {
  auto deltas = deltas_for_biases_;

  for (size_t row = 0; row < deltas.back().GetRows(); ++row) {
    auto neuron_value = pre_last_layer_neuron_->outputs[row]->val;
    double exp_res = 0.0;
    if (row == static_cast<size_t>(expected_result - 97)) exp_res = 1.0;
    deltas.back()(row, 0) =
        Sigmoid::SigmoidDerivative(neuron_value) * (neuron_value - exp_res);
  }
  auto cur_layer = pre_last_layer_neuron_;

  for (size_t layer = hidden_layers_count_; layer > 0; --layer) {
    deltas[layer - 1] = weights[layer].Transpose() * deltas[layer];
    for (size_t row = 0; row < deltas[layer - 1].GetRows(); ++row) {
      auto neuron_value = cur_layer->inputs.front()->outputs[row]->val;
      deltas[layer - 1](row, 0) *= Sigmoid::SigmoidDerivative(neuron_value);
    }
    cur_layer = cur_layer->inputs.front();
  }

  for (size_t layer = 0; layer < hidden_layers_count_ + 1; ++layer) {
    deltas_for_biases_[layer] += deltas[layer];
    for (size_t row = 0; row < deltas_for_weights_[layer].GetRows(); ++row)
      for (size_t col = 0; col < deltas_for_weights_[layer].GetCols(); ++col) {
        deltas_for_weights_[layer](row, col) +=
            deltas[layer](row, 0) *
            cur_layer->outputs.front()->inputs[col]->val;
      }
    cur_layer = cur_layer->outputs.front();
  }
}

double GraphLayers::TotalCost(unsigned char expected_result) const {
  double sum = 0;
  for (size_t row = 0; row < kOutputNeuronsCount; ++row) {
    double value = pre_last_layer_neuron_->outputs[row]->val;
    if (row == static_cast<size_t>(expected_result - 97)) {
      sum += (1.0 - value) * (1.0 - value);
    } else {
      sum += (0.0 - value) * (0.0 - value);
    }
  }
  return sum / 2.0;
}

}  // namespace s21