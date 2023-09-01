#include "emnist.h"

namespace s21 {
std::vector<Emnist::Dataset> Emnist::LoadDataset(
    const std::string& pathDataset, const std::string& pathMapping) {
  std::map<uint8_t, std::pair<char, char>> mapping;
  std::vector<Dataset> dataset;
  std::ifstream fDataset;
  std::ifstream fMapping;

  fDataset.open(pathDataset);
  fMapping.open(pathMapping);
  if (fDataset.is_open() && fMapping.is_open()) {
    std::string line;
    while (std::getline(fMapping, line).good()) {
      uint8_t key =
          std::atoi(std::strtok(const_cast<char*>(line.c_str()), " "));
      char upperCaseLetter = std::atoi(std::strtok(nullptr, " "));
      char lowerCaseLetter = std::atoi(std::strtok(nullptr, " "));
      mapping[key] = {upperCaseLetter, lowerCaseLetter};
    }

    Dataset data;
    while (std::getline(fDataset, line).good()) {
      uint8_t key =
          std::atoi(std::strtok(const_cast<char*>(line.c_str()), ","));
      auto it = mapping.find(key);
      if (it != mapping.end()) {
        char* next;
        int count = 0;
        data.upperCaseLetter = it->second.first;
        data.lowerCaseLetter = it->second.second;
        while ((next = std::strtok(nullptr, ","))) {
          data.imageData(count++, 0) = std::atoi(next) / 255.0;
        }
      }
      dataset.push_back(data);
    }
  }

  if (fDataset.is_open()) {
    fDataset.close();
  }

  if (fMapping.is_open()) {
    fMapping.close();
  }

  return dataset;
}
}  // namespace s21