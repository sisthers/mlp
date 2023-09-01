#ifndef CPP7_MLP_MODEL_MATRIX_H_
#define CPP7_MLP_MODEL_MATRIX_H_

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>

namespace s21 {
template <class T>
class S21Matrix {
 public:
  S21Matrix();
  explicit S21Matrix(size_t rows, size_t cols);
  explicit S21Matrix(size_t dimension);
  S21Matrix(const S21Matrix& other);
  S21Matrix(S21Matrix&& other) noexcept;
  ~S21Matrix();

  bool EqMatrix(const S21Matrix& other) const;
  void SumMatrix(const S21Matrix& other);
  void SubMatrix(const S21Matrix& other);
  void MulNumber(double num);
  void MulMatrix(const S21Matrix& other);
  void UseFunction(std::function<T(T)> function);
  T Determinant() const;
  S21Matrix Transpose() const;
  S21Matrix CalcComplements() const;
  S21Matrix InverseMatrix() const;

  S21Matrix operator+(const S21Matrix& other) const;
  S21Matrix operator+() const;
  S21Matrix operator-(const S21Matrix& other) const;
  S21Matrix operator-() const;
  S21Matrix operator*(const S21Matrix& other) const;
  S21Matrix operator*(double num) const;
  template <class F>
  friend S21Matrix<F> operator*(const double num, const S21Matrix<F>& other);
  S21Matrix& operator+=(const S21Matrix& other);
  S21Matrix& operator-=(const S21Matrix& other);
  S21Matrix& operator*=(const S21Matrix& other);
  S21Matrix& operator*=(double num);
  S21Matrix& operator=(const S21Matrix& other);
  S21Matrix& operator=(S21Matrix&& other) noexcept;
  bool operator==(const S21Matrix& other) const;
  T& operator()(size_t row, size_t col);
  T operator()(size_t row, size_t col) const;

  void SetRows(size_t rows);
  size_t GetRows() const noexcept;
  void SetCols(size_t cols);
  size_t GetCols() const noexcept;

 private:
  T CalcMinor() const;
  size_t rows_;
  size_t cols_;
  T* matrix_;
};

template <class T>
S21Matrix<T>::S21Matrix() : rows_(3), cols_(3) {
  matrix_ = new T[rows_ * cols_]();
}

template <class T>
S21Matrix<T>::S21Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
  if (rows == 0 || cols == 0)
    throw std::out_of_range("Number of rows or columns is equel to zero");
  matrix_ = new T[rows_ * cols_]();
}

template <class T>
S21Matrix<T>::S21Matrix(size_t dimension) : rows_(dimension), cols_(dimension) {
  if (rows_ == 0) throw std::out_of_range("Dimension is equel to zero");
  matrix_ = new T[rows_ * cols_]();
}

template <class T>
S21Matrix<T>::S21Matrix(const S21Matrix& other)
    : rows_(other.rows_), cols_(other.cols_) {
  matrix_ = new T[rows_ * cols_]();
  for (size_t row = 0; row < rows_ * cols_; ++row)
    matrix_[row] = other.matrix_[row];
}

template <class T>
S21Matrix<T>::S21Matrix(S21Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_) {
  matrix_ = other.matrix_;
  other.matrix_ = nullptr;
  other.rows_ = 0;
  other.cols_ = 0;
}

template <class T>
S21Matrix<T>::~S21Matrix() {
  delete[] matrix_;
}

template <class T>
bool S21Matrix<T>::EqMatrix(const S21Matrix& other) const {
  bool result = rows_ == other.rows_ && cols_ == other.cols_;
  for (size_t row = 0; row < rows_ && result; ++row)
    for (size_t col = 0; col < cols_ && result; ++col)
      if (fabs((*this)(row, col) - other(row, col)) > 1e-6) result = false;
  return result;
}

template <class T>
void S21Matrix<T>::SumMatrix(const S21Matrix& other) {
  if (rows_ != other.rows_ || cols_ != other.cols_)
    throw std::out_of_range("Matrices have different dimensions");
  for (size_t row = 0; row < rows_; ++row)
    for (size_t col = 0; col < cols_; ++col)
      (*this)(row, col) += other(row, col);
}

template <class T>
void S21Matrix<T>::SubMatrix(const S21Matrix& other) {
  SumMatrix(-other);
}

template <class T>
void S21Matrix<T>::MulNumber(double num) {
  for (size_t row = 0; row < rows_; ++row)
    for (size_t col = 0; col < cols_; ++col) (*this)(row, col) *= num;
}

template <class T>
void S21Matrix<T>::MulMatrix(const S21Matrix& other) {
  if (cols_ != other.rows_)
    throw std::out_of_range(
        "Number of columns of ther first matrix is not equal to number of rows "
        "of the second matrix");
  S21Matrix<T> result(rows_, other.cols_);
  for (size_t row = 0; row < rows_; ++row)
    for (size_t col = 0; col < other.cols_; ++col) {
      T sum = T();
      for (size_t x = 0; x < cols_; ++x) sum += (*this)(row, x) * other(x, col);
      result(row, col) = sum;
    }
  *this = std::move(result);
}

template <class T>
void S21Matrix<T>::UseFunction(std::function<T(T)> function) {
  for (size_t row = 0; row < rows_; ++row)
    for (size_t col = 0; col < cols_; ++col)
      (*this)(row, col) = function((*this)(row, col));
}

template <class T>
S21Matrix<T> S21Matrix<T>::Transpose() const {
  S21Matrix result(cols_, rows_);
  for (size_t row = 0; row < rows_; ++row)
    for (size_t col = 0; col < cols_; ++col)
      result(col, row) = (*this)(row, col);
  return result;
}

template <class T>
S21Matrix<T> S21Matrix<T>::CalcComplements() const {
  if (rows_ != cols_) throw std::out_of_range("Matrix is not quadratic");
  S21Matrix<T> result(rows_, rows_);
  for (size_t row = 0; row < rows_; ++row)
    for (size_t col = 0; col < rows_; ++col) {
      S21Matrix<T> minor(rows_ - 1, rows_ - 1);
      for (size_t rowM = 0; rowM < rows_; ++rowM)
        for (size_t colM = 0; colM < cols_; ++colM)
          if (rowM != row && colM != col)
            minor(rowM > row ? rowM - 1 : rowM, colM > col ? colM - 1 : colM) =
                (*this)(rowM, colM);
      result(row, col) =
          (row + col) % 2 == 0 ? minor.CalcMinor() : -minor.CalcMinor();
    }
  return result;
}

template <class T>
T S21Matrix<T>::Determinant() const {
  if (rows_ != cols_) throw std::out_of_range("Matrix is not quadratic");
  T result = T();
  if (rows_ == 1 || rows_ == 2) {
    result = CalcMinor();
  } else {
    S21Matrix<T> complements = CalcComplements();
    for (size_t col = 0; col < rows_; ++col)
      result += (*this)(0, col) * complements(0, col);
  }
  return result;
}

template <class T>
S21Matrix<T> S21Matrix<T>::InverseMatrix() const {
  if (rows_ != cols_) throw std::out_of_range("Matrix is not quadratic");
  T det = Determinant();
  if (fabs(det - 0) <= 1e-6)
    throw std::out_of_range("Determinant is equel to zero");
  S21Matrix<T> result(rows_, rows_);
  if (rows_ == 1) {
    result(0, 0) = (double)1 / (*this)(0, 0);
  } else {
    S21Matrix<T> complements = CalcComplements();
    S21Matrix<T> transposed = complements.Transpose();
    transposed.MulNumber(1.0 / det);
    result = transposed;
  }
  return result;
}

template <class T>
T S21Matrix<T>::CalcMinor() const {
  T result = T();
  if (rows_ == 1)
    result = (*this)(0, 0);
  else if (rows_ == 2)
    result = (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
  else {
    for (size_t col = 0; col < rows_; ++col) {
      S21Matrix<T> minor(rows_ - 1, rows_ - 1);
      for (size_t rowM = 0; rowM < rows_; ++rowM)
        for (size_t colM = 0; colM < rows_; ++colM) {
          if (rowM != 0 && colM != col)
            minor(rowM > 0 ? rowM - 1 : rowM, colM > col ? colM - 1 : colM) =
                (*this)(rowM, colM);
        }
      result += (col % 2 == 0 ? minor.CalcMinor() : -minor.CalcMinor()) *
                (*this)(0, col);
    }
  }
  return result;
}

template <class T>
S21Matrix<T> S21Matrix<T>::operator+(const S21Matrix& other) const {
  S21Matrix<T> result(*this);
  result.SumMatrix(other);
  return result;
}

template <class T>
S21Matrix<T> S21Matrix<T>::operator+() const {
  return *this;
}

template <class T>
S21Matrix<T> S21Matrix<T>::operator-(const S21Matrix& other) const {
  S21Matrix<T> result(*this);
  result.SubMatrix(other);
  return result;
}

template <class T>
S21Matrix<T> S21Matrix<T>::operator-() const {
  S21Matrix<T> result = *this * -1;
  return result;
}

template <class T>
S21Matrix<T> S21Matrix<T>::operator*(const S21Matrix& other) const {
  S21Matrix<T> result(*this);
  result.MulMatrix(other);
  return result;
}

template <class T>
S21Matrix<T> S21Matrix<T>::operator*(double num) const {
  S21Matrix<T> result(*this);
  result.MulNumber(num);
  return result;
}

template <class T>
S21Matrix<T> operator*(double num, const S21Matrix<T>& other) {
  S21Matrix<T> result(other);
  result.MulNumber(num);
  return result;
}

template <class T>
S21Matrix<T>& S21Matrix<T>::operator+=(const S21Matrix& other) {
  SumMatrix(other);
  return *this;
}

template <class T>
S21Matrix<T>& S21Matrix<T>::operator-=(const S21Matrix& other) {
  SubMatrix(other);
  return *this;
}

template <class T>
S21Matrix<T>& S21Matrix<T>::operator*=(const S21Matrix& other) {
  MulMatrix(other);
  return *this;
}

template <class T>
S21Matrix<T>& S21Matrix<T>::operator*=(const double num) {
  MulNumber(num);
  return *this;
}

template <class T>
S21Matrix<T>& S21Matrix<T>::operator=(const S21Matrix& other) {
  if (&other == this) return *this;
  delete[] matrix_;
  matrix_ = new T[other.rows_ * other.cols_];
  for (size_t row = 0; row < other.rows_ * other.cols_; ++row)
    matrix_[row] = other.matrix_[row];
  rows_ = other.rows_;
  cols_ = other.cols_;
  return *this;
}

template <class T>
S21Matrix<T>& S21Matrix<T>::operator=(S21Matrix&& other) noexcept {
  std::swap(matrix_, other.matrix_);
  std::swap(rows_, other.rows_);
  std::swap(cols_, other.cols_);
  return *this;
}

template <class T>
bool S21Matrix<T>::operator==(const S21Matrix& other) const {
  return EqMatrix(other);
}

template <class T>
T& S21Matrix<T>::operator()(size_t row, size_t col) {
  if (row >= rows_ || col >= cols_)
    throw std::out_of_range("Index is outside the matrix");
  return matrix_[cols_ * row + col];
}
template <class T>
T S21Matrix<T>::operator()(size_t row, size_t col) const {
  if (row >= rows_ || col >= cols_)
    throw std::out_of_range("Index is outside the matrix");
  return matrix_[cols_ * row + col];
}

template <class T>
void S21Matrix<T>::SetRows(size_t rows) {
  if (rows == 0) throw std::out_of_range("Index is equel to zero");
  if (rows_ == rows) return;
  T* temp = new T[rows * cols_]();
  for (size_t row = 0; row < rows; ++row)
    for (size_t col = 0; col < cols_; ++col)
      temp[row * cols_ + col] = row < rows_ ? matrix_[row * cols_ + col] : T();
  delete[] matrix_;
  matrix_ = temp;
  rows_ = rows;
}

template <class T>
size_t S21Matrix<T>::GetRows() const noexcept {
  return rows_;
}

template <class T>
void S21Matrix<T>::SetCols(size_t cols) {
  if (cols == 0) throw std::out_of_range("Index is equel to zero");
  if (cols_ == cols) return;
  T* temp = new T[rows_ * cols]();
  for (size_t row = 0; row < rows_; ++row)
    for (size_t col = 0; col < cols; ++col)
      temp[row * cols + col] = col < cols_ ? matrix_[row * cols_ + col] : T();
  delete[] matrix_;
  matrix_ = temp;
  cols_ = cols;
}

template <class T>
size_t S21Matrix<T>::GetCols() const noexcept {
  return cols_;
}

}  // namespace s21

#endif  // CPP7_MLP_MODEL_MATRIX_H_