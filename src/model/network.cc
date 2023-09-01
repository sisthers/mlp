#include "network.h"

namespace s21 {
Network::~Network() { delete layers; }

Network::Network(NetworkImplementation network_implementation,
                 size_t hidden_layers_count)
    : network_implementation_(network_implementation),
      hidden_layers_count_(hidden_layers_count) {
  if (network_implementation == NetworkImplementation::kGraphForm)
    layers = new GraphLayers(hidden_layers_count);
  else
    layers = new MatrixLayers(hidden_layers_count);

  weights_.reserve(hidden_layers_count + 1);
  biases_.reserve(hidden_layers_count + 1);

  weights_.push_back(S21Matrix<double>(layers->kNeuronsOnHiddenLayerCount,
                                       layers->kInputNeuronsCount));
  for (size_t i = 0; i < hidden_layers_count; ++i) {
    if (i != hidden_layers_count - 1)
      weights_.push_back(S21Matrix<double>(layers->kNeuronsOnHiddenLayerCount));
    biases_.push_back(S21Matrix<double>(layers->kNeuronsOnHiddenLayerCount, 1));
  }
  weights_.push_back(S21Matrix<double>(layers->kOutputNeuronsCount,
                                       layers->kNeuronsOnHiddenLayerCount));
  biases_.push_back(S21Matrix<double>(layers->kOutputNeuronsCount, 1));

  std::random_device device;
  random_gen_.seed(device());
}

Network::TestResults::TestResults(double average_accuracy, double precision,
                                  double recall, double f_measure,
                                  double total_time, double average_loss)
    : average_accuracy(average_accuracy),
      precision(precision),
      recall(recall),
      f_measure(f_measure),
      total_time(total_time),
      average_loss(average_loss) {}

void Network::SaveWeightsAndBiases(const std::string &file_name) const {
  std::ofstream file_stream;
  file_stream.open(file_name);
  for (size_t i = 0; i < weights_.size(); ++i) {
    for (size_t row = 0; row < weights_[i].GetRows(); ++row) {
      for (size_t col = 0; col < weights_[i].GetCols(); ++col) {
        file_stream << weights_[i](row, col);
        if (col != weights_[i].GetCols() - 1) file_stream << ' ';
      }
      file_stream << '\n';
    }
  }

  for (size_t i = 0; i < biases_.size(); ++i) {
    for (size_t row = 0; row < biases_[i].GetRows(); ++row) {
      file_stream << biases_[i](row, 0);
      file_stream << '\n';
    }
  }
  if (file_stream.is_open()) file_stream.close();
}

bool Network::LoadWeightsAndBiases(const std::string &file_name) {
  std::ifstream file_stream;
  file_stream.open(file_name);
  if (!file_stream.is_open()) return false;

  for (size_t i = 0; i < weights_.size(); ++i) {
    for (size_t row = 0; row < weights_[i].GetRows(); ++row) {
      for (size_t col = 0; col < weights_[i].GetCols(); ++col) {
        if (file_stream.eof()) {
          file_stream.close();
          return false;
        }
        file_stream >> weights_[i](row, col);
      }
    }
  }

  for (size_t i = 0; i < biases_.size(); ++i) {
    for (size_t row = 0; row < biases_[i].GetRows(); ++row) {
      if (file_stream.eof()) {
        file_stream.close();
        return false;
      }
      file_stream >> biases_[i](row, 0);
    }
  }

  if (file_stream.is_open()) file_stream.close();
  trained = true;
  return true;
}

void Network::ChangeImplenetation(
    NetworkImplementation network_implementation) {
  if (network_implementation == network_implementation_) return;
  delete layers;
  if (network_implementation == NetworkImplementation::kGraphForm)
    layers = new GraphLayers(hidden_layers_count_);
  else
    layers = new MatrixLayers(hidden_layers_count_);
  network_implementation_ = network_implementation;
}

void Network::ChangeHiddenLayersNumber(size_t number) {
  if (number == hidden_layers_count_) return;
  layers->ChangeNumberOfHiddenLayers(number);
  if (number > hidden_layers_count_) {
    weights_.insert(weights_.end() - 1, number - hidden_layers_count_,
                    S21Matrix<double>(layers->kNeuronsOnHiddenLayerCount));
    biases_.insert(biases_.end() - 1, number - hidden_layers_count_,
                   S21Matrix<double>(layers->kNeuronsOnHiddenLayerCount, 1));
  } else {
    weights_.erase(weights_.begin() + 1,
                   weights_.begin() + 1 + hidden_layers_count_ - number);
    biases_.erase(biases_.begin(),
                  biases_.begin() + hidden_layers_count_ - number);
  }
  hidden_layers_count_ = number;
  trained = false;
}

char Network::GetPrediction(const S21Matrix<double> &image) const {
  if (!trained) throw std::runtime_error("Network is not trained");
  layers->FeedForward(image, weights_, biases_);
  return 97 + layers->GetMaxOutputIndex();
}

typename Network::TestResults Network::RunTests(const std::string &data_path,
                                                const std::string &mapping_path,
                                                double sample_part) {
  if (sample_part < 0.0 || sample_part > 1.0)
    throw std::runtime_error(
        "Sample part should be a number between 0.0 and 1.0");
  auto samples = Emnist::LoadDataset(data_path, mapping_path);

  std::shuffle(samples.begin(), samples.end(), random_gen_);
  size_t last_el_index = samples.size() * sample_part;
  auto result = RunTests(samples.begin(), samples.begin() + last_el_index);
  return result;
}

std::vector<Network::TestResults> Network::StartLearning(
    const std::string &data_path, const std::string &test_path,
    const std::string &mapping_path, size_t epochs_count) {
  if (epochs_count == 0) throw std::runtime_error("Invalid number of epochs");
  trained = true;
  InitWeights();
  std::vector<TestResults> result;
  result.reserve(epochs_count);

  auto samples = Emnist::LoadDataset(data_path, mapping_path);

  for (size_t epoch = 0; epoch < epochs_count; ++epoch) {
    std::shuffle(samples.begin(), samples.end(), random_gen_);
    auto losses = Train(samples, epoch, epochs_count);
    auto test_result = RunTests(test_path, mapping_path, 1);
    test_result.average_loss =
        std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();
    result.push_back(test_result);
  }
  return result;
}

std::vector<Network::TestResults> Network::StartLearningWithCrossValidation(
    const std::string &data_path, const std::string &mapping_path, size_t k) {
  if (k < 5 || k > 10) throw std::runtime_error("Invalid number of gropus");
  trained = true;
  InitWeights();
  std::vector<TestResults> result;
  result.reserve(k);
  auto samples = Emnist::LoadDataset(data_path, mapping_path);
  std::shuffle(samples.begin(), samples.end(), random_gen_);
  size_t group_size = samples.size() / k;
  for (size_t group = 0; group < k; ++group) {
    std::vector<Emnist::Dataset> train_samples;
    std::vector<Emnist::Dataset> test_samples;
    if (group != k - 1) {
      test_samples.reserve(group_size);
      test_samples.insert(test_samples.end(),
                          samples.begin() + group * group_size,
                          samples.begin() + (group + 1) * group_size);
      train_samples.reserve(samples.size() - test_samples.size());
      train_samples.insert(train_samples.end(), samples.begin(),
                           samples.begin() + group * group_size);
      train_samples.insert(train_samples.end(),
                           samples.begin() + (group + 1) * group_size,
                           samples.end());
    } else {
      test_samples.reserve(samples.size() - group * group_size);
      test_samples.insert(test_samples.end(),
                          samples.begin() + group * group_size, samples.end());
      train_samples.reserve(samples.size() - test_samples.size());
      train_samples.insert(train_samples.end(), samples.begin(),
                           samples.begin() + group * group_size);
    }
    auto losses = Train(train_samples, group, k);
    auto test_result = RunTests(test_samples.begin(), test_samples.end());
    test_result.average_loss =
        std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();
    result.push_back(test_result);
  }
  return result;
}

size_t Network::GetMiniBatchSize() const noexcept {
  return layers->GetMiniBatchSize();
}

void Network::SetMiniBatchSize(size_t size) { layers->SetMiniBatchSize(size); }

void Network::InitWeights() {
  for (size_t i = 0; i < weights_.size(); ++i) {
    std::uniform_real_distribution<> dist_w(
        -(sqrt(6) / sqrt(weights_[i].GetCols() + weights_[i].GetRows())),
        sqrt(6) / sqrt(weights_[i].GetCols() + weights_[i].GetRows()));
    std::uniform_real_distribution<> dist_b(
        -(sqrt(6) / sqrt(biases_[i].GetCols() + biases_[i].GetRows())),
        sqrt(6) / sqrt(biases_[i].GetCols() + biases_[i].GetRows()));
    for (size_t row = 0; row < weights_[i].GetRows(); ++row) {
      biases_[i](row, 0) = dist_b(random_gen_);
      for (size_t col = 0; col < weights_[i].GetCols(); ++col)
        weights_[i](row, col) = dist_w(random_gen_);
    }
  }
}

Network::TestResults Network::RunTests(
    std::vector<Emnist::Dataset>::iterator start,
    std::vector<Emnist::Dataset>::iterator end) {
  S21Matrix<size_t> confusion_matrix(layers->kOutputNeuronsCount);
  size_t correct_guesses = 0;

  auto clock_start = std::chrono::high_resolution_clock::now();
  for (auto sample = start; sample < end; ++sample) {
    auto prediction = GetPrediction(sample->imageData);
    auto expected_result = sample->lowerCaseLetter;
    if (prediction == expected_result) ++correct_guesses;
    ++confusion_matrix(prediction - 97, expected_result - 97);
  }
  auto clock_end = std::chrono::high_resolution_clock::now();

  double precision = 0;
  double recall = 0;

  for (size_t i = 0; i < layers->kOutputNeuronsCount; ++i) {
    double diag_el = confusion_matrix(i, i);
    double col_sum = 0;
    double row_sum = 0;
    for (size_t j = 0; j < layers->kOutputNeuronsCount; ++j) {
      row_sum += confusion_matrix(i, j);
      col_sum += confusion_matrix(j, i);
    }
    if (fabs(row_sum - 0.0) > 1e-6) precision += diag_el / row_sum;
    if (fabs(col_sum - 0.0) > 1e-6) recall += diag_el / col_sum;
  }
  precision /= layers->kOutputNeuronsCount;
  recall /= layers->kOutputNeuronsCount;
  double f_measure = 2 * (precision * recall) / (precision + recall);

  Network::TestResults test_results(
      correct_guesses / static_cast<double>(std::distance(start, end)),
      precision, recall, f_measure,
      std::chrono::duration_cast<std::chrono::seconds>(clock_end - clock_start)
          .count());
  return test_results;
}

std::vector<double> Network::Train(std::vector<Emnist::Dataset> &samples,
                                   size_t iteration, size_t iterations_count) {
  size_t mini_batch_size = layers->GetMiniBatchSize();
  std::vector<double> losses;
  size_t samples_size = samples.size();
  losses.reserve(samples_size);

  size_t sample = 0;
  while (sample < samples_size) {
    for (size_t mini_batch_sample = 0;
         sample < samples_size && mini_batch_sample < mini_batch_size;
         ++mini_batch_sample, ++sample) {
      layers->FeedForward(samples[sample].imageData, weights_, biases_);
      layers->BackPropogation(samples[sample].lowerCaseLetter, weights_);
      losses.push_back(layers->TotalCost(samples[sample].lowerCaseLetter));
    }
    layers->UpdateWeights(
        weights_, biases_,
        0.99 * exp(-(static_cast<double>(iteration) / iterations_count)));
    layers->ResetDeltas();
  }
  return losses;
}

}  // namespace s21