#ifndef CPP7_MLP_MODEL_SIGMOID_H_
#define CPP7_MLP_MODEL_SIGMOID_H_

#include <cmath>

namespace s21 {
class Sigmoid {
 public:
  static double SigmoidFunction(double x);
  static double SigmoidDerivative(double x);
};

}  // namespace s21
#endif  // CPP7_MLP_MODEL_SIGMOID_H_
