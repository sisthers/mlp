#include "sigmoid.h"

#include <iostream>
namespace s21 {
double Sigmoid::SigmoidFunction(double x) { return 1.0 / (1.0 + exp(-x)); }

double Sigmoid::SigmoidDerivative(double x) { return x * (1.0 - x); }
}  // namespace s21