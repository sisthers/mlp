#ifndef CPP7_MLP_VIEW_VIEW_H_
#define CPP7_MLP_VIEW_VIEW_H_

#include <QChart>
#include <QChartView>
#include <QDoubleValidator>
#include <QFileDialog>
#include <QImage>
#include <QLineSeries>
#include <QLogValueAxis>
#include <QMessageBox>
#include <QPainter>
#include <QTimer>
#include <QValueAxis>
#include <QWidget>
#include <QtCharts>
#include <atomic>
#include <thread>
#include <vector>

#include "controller.h"
#include "draw.h"
#include "s21_matrix.h"
#include "spinner.h"
#include "ui_view.h"

namespace s21 {

class View : public QWidget {
  friend Draw;

  Q_OBJECT

 public:
  View(QWidget* parent = nullptr);
  ~View();

 private slots:
  void LoadImage();
  void DrawImage();
  void Classify();
  void StartLearning();
  void StartLearningValidation();
  void LoadPerceptron();
  void SavePerceptron();
  void RunTest();
  void UpdateChart();
  void ChangeLayersNumber(int index);

 private:
  Ui::ViewClass ui_;
  QImage image_;
  Draw* draw_;
  Controller controller_;
  QChartView* chart_view_;
  Spinner* spinner_;
  QTimer* timer_;

  std::atomic<bool> chart_update_;
  std::vector<Network::TestResults> test_results_;

  bool eventFilter(QObject* watched, QEvent* event);

  void StartLearningTh(const std::string& data_path,
                       const std::string& test_path,
                       const std::string& mapping_path, size_t epochs_count);
  void StartLearningWithCrossValidationTh(const std::string& data_path,
                                          const std::string& mapping_path,
                                          size_t k);
};

}  // namespace s21

#endif  // CPP7_MLP_VIEW_VIEW_H_
