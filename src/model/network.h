#ifndef CPP7_MLP_MODEL_NETWORK_H_
#define CPP7_MLP_MODEL_NETWORK_H_

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

#include "emnist.h"
#include "layers.h"
#include "s21_matrix.h"

namespace s21 {

class Network {
 public:
  struct TestResults {
   public:
    TestResults(double average_accuracy, double precision, double recall,
                double f_measure, double total_time, double average_error = 0);

    double average_accuracy;
    double precision;
    double recall;
    double f_measure;
    double total_time;
    double average_loss;
  };

  enum class NetworkImplementation { kMatrixForm = 0, kGraphForm = 1 };

  explicit Network(NetworkImplementation network_implementationl,
                   size_t hidden_layers_count);
  Network(const Network& network) = delete;
  Network(Network&& network) = delete;
  Network& operator=(const Network& network) = delete;
  Network& operator=(Network&& network) = delete;
  ~Network();

  void SaveWeightsAndBiases(const std::string& file_name) const;
  bool LoadWeightsAndBiases(const std::string& file_name);
  void ChangeImplenetation(NetworkImplementation network_implementation);
  void ChangeHiddenLayersNumber(size_t number);
  char GetPrediction(const S21Matrix<double>& image) const;
  TestResults RunTests(const std::string& data_path,
                       const std::string& mapping_path, double sample_part);
  std::vector<TestResults> StartLearning(const std::string& data_path,
                                         const std::string& test_path,
                                         const std::string& mapping_path,
                                         size_t epochs_count);
  std::vector<TestResults> StartLearningWithCrossValidation(
      const std::string& data_path, const std::string& mapping_path, size_t k);
  size_t GetMiniBatchSize() const noexcept;
  void SetMiniBatchSize(size_t size);

 private:
  void InitWeights();
  TestResults RunTests(std::vector<Emnist::Dataset>::iterator start,
                       std::vector<Emnist::Dataset>::iterator end);
  std::vector<double> Train(std::vector<Emnist::Dataset>& samples,
                            size_t iteration, size_t iterations_count);

  std::mt19937 random_gen_;
  NetworkImplementation network_implementation_;
  Layers* layers;
  std::vector<S21Matrix<double>> weights_;
  std::vector<S21Matrix<double>> biases_;
  size_t hidden_layers_count_;
  bool trained = false;
};
}  // namespace s21

#endif  // CPP7_MLP_MODEL_NETWORK_H_
