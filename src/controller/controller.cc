#include "controller.h"

namespace s21 {

void Controller::SaveWeightsAndBiases(const std::string& file_name) {
  network_.SaveWeightsAndBiases(file_name);
}

bool Controller::LoadWeightsAndBiases(const std::string& file_name) {
  return network_.LoadWeightsAndBiases(file_name);
}

void Controller::ChangeImplenetation(
    Network::NetworkImplementation network_implementation) {
  network_.ChangeImplenetation(network_implementation);
}

void Controller::ChangeHiddenLayersNumber(size_t number) {
  network_.ChangeHiddenLayersNumber(number);
}

char Controller::GetPrediction(const S21Matrix<double>& image) const {
  return network_.GetPrediction(image);
}

void Controller::SetMBSize(size_t size) { network_.SetMiniBatchSize(size); }

std::vector<Network::TestResults> Controller::StartLearning(
    const std::string& data_path, const std::string& test_path,
    const std::string& mapping_path, size_t epochs_count) {
  return network_.StartLearning(data_path, test_path, mapping_path,
                                epochs_count);
}

std::vector<Network::TestResults> Controller::StartLearningWithCrossValidation(
    const std::string& data_path, const std::string& mapping_path, size_t k) {
  return network_.StartLearningWithCrossValidation(data_path, mapping_path, k);
}

Network::TestResults Controller::RunTests(const std::string& data_path,
                                          const std::string& mapping_path,
                                          double sample_part) {
  return network_.RunTests(data_path, mapping_path, sample_part);
}

}  // namespace s21
