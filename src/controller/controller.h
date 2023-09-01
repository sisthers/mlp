#ifndef CPP7_MLP_CONTROLLER_CONTROLLER_H_
#define CPP7_MLP_CONTROLLER_CONTROLLER_H_

#include "network.h"

namespace s21 {

class Controller {
 public:
  Controller() : network_(Network::NetworkImplementation::kMatrixForm, 2) {}
  void SaveWeightsAndBiases(const std::string& file_name);
  bool LoadWeightsAndBiases(const std::string& file_name);
  void ChangeImplenetation(
      Network::NetworkImplementation network_implementation);
  void ChangeHiddenLayersNumber(size_t number);
  char GetPrediction(const S21Matrix<double>& image) const;
  void SetMBSize(size_t size);
  std::vector<Network::TestResults> StartLearning(
      const std::string& data_path, const std::string& test_path,
      const std::string& mapping_path, size_t epochs_count);
  std::vector<Network::TestResults> StartLearningWithCrossValidation(
      const std::string& dataPath, const std::string& mapping_path, size_t k);
  Network::TestResults RunTests(const std::string& data_path,
                                const std::string& mapping_path,
                                double sample_part);

 private:
  Network network_;
};

}  // namespace s21

#endif  // CPP7_MLP_CONTROLLER_CONTROLLER_H_
