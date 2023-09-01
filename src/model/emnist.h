#ifndef CPP7_MLP_MODEL_EMNIST_H_
#define CPP7_MLP_MODEL_EMNIST_H_

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "s21_matrix.h"

namespace s21 {
class Emnist {
 public:
  struct Dataset {
   public:
    char upperCaseLetter;
    char lowerCaseLetter;
    S21Matrix<double> imageData = S21Matrix<double>(784, 1);
  };

  static std::vector<Dataset> LoadDataset(const std::string& pathDataset,
                                          const std::string& pathMapping);
};
}  // namespace s21

#endif  // CPP7_MLP_MODEL_EMNIST_H_