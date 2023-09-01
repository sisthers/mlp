#ifndef CPP7_MLP_MODEL_LAYERS_H_
#define CPP7_MLP_MODEL_LAYERS_H_

#include <random>
#include <vector>

#include "s21_matrix.h"
#include "sigmoid.h"

namespace s21 {
class Layers {
 public:
  explicit Layers(size_t hidden_layers_count);
  Layers(const Layers& layers) = delete;
  Layers(Layers&& layers) = delete;
  Layers& operator=(const Layers& layers) = delete;
  Layers& operator=(Layers&& layers) = delete;
  virtual ~Layers();

  virtual void FeedForward(const S21Matrix<double>& image,
                           const std::vector<S21Matrix<double>>& weights,
                           const std::vector<S21Matrix<double>>& biases) = 0;
  virtual size_t GetMaxOutputIndex() const = 0;
  virtual void ChangeNumberOfHiddenLayers(size_t number) = 0;
  virtual void BackPropogation(unsigned char expected_result,
                               std::vector<S21Matrix<double>>& weights) = 0;
  virtual double TotalCost(unsigned char expected_result) const = 0;
  void UpdateWeights(std::vector<S21Matrix<double>>& weights,
                     std::vector<S21Matrix<double>>& biases,
                     double learning_rate);
  void ResetDeltas();
  void SetMiniBatchSize(size_t size);
  size_t GetMiniBatchSize() const noexcept;

  const size_t kNeuronsOnHiddenLayerCount = 50;
  const size_t kInputNeuronsCount = 784;
  const size_t kOutputNeuronsCount = 26;

 protected:
  size_t hidden_layers_count_;
  size_t mini_batch_size_ = 32;
  std::vector<S21Matrix<double>> deltas_for_weights_;
  std::vector<S21Matrix<double>> deltas_for_biases_;
};

class GraphLayers : public Layers {
 public:
  explicit GraphLayers(size_t hidden_layers_count);
  ~GraphLayers();

  void FeedForward(const S21Matrix<double>& image,
                   const std::vector<S21Matrix<double>>& weights,
                   const std::vector<S21Matrix<double>>& biases) override;
  size_t GetMaxOutputIndex() const noexcept override;
  void ChangeNumberOfHiddenLayers(size_t number) override;
  void BackPropogation(unsigned char expected_result,
                       std::vector<S21Matrix<double>>& weights) override;
  double TotalCost(unsigned char expected_result) const override;

 private:
  struct Neuron {
   public:
    std::vector<Neuron*> inputs;
    std::vector<Neuron*> outputs;
    double val;
    size_t layer;
  };

  Neuron* root_0_0_neuron_;
  Neuron* pre_last_layer_neuron_;
};

class MatrixLayers : public Layers {
 public:
  explicit MatrixLayers(size_t hidden_layers_count);
  ~MatrixLayers();

  void FeedForward(const S21Matrix<double>& image,
                   const std::vector<S21Matrix<double>>& weights,
                   const std::vector<S21Matrix<double>>& biases) override;
  size_t GetMaxOutputIndex() const noexcept override;
  void ChangeNumberOfHiddenLayers(size_t number) override;
  void BackPropogation(unsigned char expected_result,
                       std::vector<S21Matrix<double>>& weights) override;
  double TotalCost(unsigned char expected_result) const override;

 private:
  std::vector<S21Matrix<double>> neurons_;
};
}  // namespace s21
#endif  // CPP7_MLP_MODEL_LAYERS_H_