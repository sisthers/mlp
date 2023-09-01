#include "view.h"

namespace s21 {

View::View(QWidget* parent) : QWidget(parent), chart_view_(nullptr) {
  ui_.setupUi(this);
  connect(ui_.loadImage, &QPushButton::clicked, this, &View::LoadImage);
  connect(ui_.drawImage, &QPushButton::clicked, this, &View::DrawImage);
  connect(ui_.classify, &QPushButton::clicked, this, &View::Classify);
  connect(ui_.startLearning, &QPushButton::clicked, this, &View::StartLearning);
  connect(ui_.startLearningValidation, &QPushButton::clicked, this,
          &View::StartLearningValidation);
  connect(ui_.loadPerceptron, &QPushButton::clicked, this,
          &View::LoadPerceptron);
  connect(ui_.savePerceptron, &QPushButton::clicked, this,
          &View::SavePerceptron);
  connect(ui_.runTest, &QPushButton::clicked, this, &View::RunTest);
  connect(ui_.hiddenLayers, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &View::ChangeLayersNumber);

  ui_.imgArea->installEventFilter(this);
  draw_ = new Draw(this);

  QLocale lo(QLocale::C);
  lo.setNumberOptions(QLocale::RejectGroupSeparator);
  auto dval = new QDoubleValidator(0, 1, 2, this);
  auto ival = new QIntValidator(this);
  dval->setLocale(lo);
  ival->setLocale(lo);
  ui_.samplePart->setValidator(dval);
  ui_.samplePart->setText("0.5");
  ui_.perceptronMode->addItems({"Matrix ", "Graph "});
  ui_.hiddenLayers->addItems({"2", "3", "4", "5"});
  ui_.groupsMode->addItems({"5", "6", "7", "8", "9", "10"});
  ui_.mbSize->addItems(
      {"1", "2", "4", "8", "16", "32", "64", "128", "256", "512"});
  ui_.mbSize->setCurrentText("32");
  ui_.epochs->setValidator(ival);
  ui_.epochs->setText("1");

  spinner_ = new Spinner(this);
  spinner_->SetRoundness(70.0);
  spinner_->SetMinimumTrailOpacity(15.0);
  spinner_->SetTrailFadePercentage(70.0);
  spinner_->SetNumberOfLines(12);
  spinner_->SetLineLength(25);
  spinner_->SetLineWidth(5);
  spinner_->SetInnerRadius(50);
  spinner_->SetRevolutionsPerSecond(1);
  spinner_->SetColor(QColor(81, 4, 71));

  timer_ = new QTimer(this);
  timer_->setInterval(500);
  connect(timer_, &QTimer::timeout, this, &View::UpdateChart);
}

View::~View() {}

void View::ChangeLayersNumber(int index) {
  size_t number = index + 2;
  controller_.ChangeHiddenLayersNumber(number);
}

void View::LoadImage() {
  auto fileName = QFileDialog::getOpenFileName(this, tr("Open a file"), ".",
                                               tr("Image files (*.bmp)"));
  if (!fileName.isEmpty()) {
    image_.load(fileName);
    update();
  }
}

bool View::eventFilter(QObject* watched, QEvent* event) {
  if (watched == ui_.imgArea && event->type() == QEvent::Paint) {
    QPainter painter;
    painter.begin(ui_.imgArea);
    auto w = ui_.imgArea->width();
    auto h = ui_.imgArea->height();
    if (!image_.isNull()) {
      QPixmap pix;
      pix.convertFromImage(image_.scaled(w, h));
      painter.drawPixmap(QRect(0, 0, w, h), pix);
    } else {
      painter.setPen(QPen(Qt::black, Qt::SolidLine));
      painter.drawRect(0, 0, w - 1, h - 1);
    }
    painter.end();
  }
  return true;
}

void View::UpdateChart() {
  if (!chart_update_) return;

  timer_->stop();
  spinner_->Stop();

  if (chart_view_) {
    delete chart_view_;
  }

  chart_view_ = new QChartView();

  ui_.learningGraph->addWidget(chart_view_);

  std::vector<QLineSeries*> series;
  series.push_back(new QLineSeries());
  series.push_back(new QLineSeries());

  for (size_t i = 0; i < test_results_.size(); i++) {
    series[0]->append(i, test_results_[i].average_accuracy);
    series[1]->append(i, test_results_[i].average_loss);
  }
  series[0]->setPen(QPen(Qt::blue));
  series[1]->setPen(QPen(Qt::red));
  series[0]->setPointsVisible();
  series[1]->setPointsVisible();

  auto chart = new QChart();
  chart->addSeries(series[0]);
  chart->addSeries(series[1]);

  chart->createDefaultAxes();
  chart->legend()->hide();
  chart_view_->setChart(chart);
}

void View::DrawImage() { draw_->show(); }

void View::Classify() {
  if (image_.isNull()) return;

  auto img = image_.scaled(28, 28);

  S21Matrix<double> img_data(28 * 28, 1);
  size_t index = 0;

  for (int h = 0; h < img.height(); h++) {
    for (int w = 0; w < img.width(); w++, index++) {
      auto rgb = img.pixel(h, w);
      img_data(index, 0) = qGray(rgb) / 255.0;
    }
  }
  try {
    char symbol = controller_.GetPrediction(img_data);
    QString result = QString("Your letter is: ").append(symbol);
    ui_.result->setText(result);
  } catch (const std::exception& ex) {
    QMessageBox msgBox;
    msgBox.setText(ex.what());
    msgBox.exec();
  }
}

void View::StartLearningTh(const std::string& dataPath,
                           const std::string& testPath,
                           const std::string& mappingPath, size_t epochsCount) {
  test_results_ =
      controller_.StartLearning(dataPath, testPath, mappingPath, epochsCount);
  chart_update_ = true;
}

void View::StartLearning() {
  std::string data_path;
  std::string test_path;
  std::string mapping_path;

  QString file_name;
  file_name = QFileDialog::getOpenFileName(this, tr("Open train dataset"), ".",
                                           tr("dataset files (*.csv)"));
  if (!file_name.isEmpty()) {
    data_path = file_name.toStdString();

    file_name = QFileDialog::getOpenFileName(this, tr("Open test dataset"), ".",
                                             tr("dataset files (*.csv)"));
    if (!file_name.isEmpty()) {
      test_path = file_name.toStdString();
      file_name = QFileDialog::getOpenFileName(
          this, tr("Open mapping file"), ".", tr("dataset files (*.txt)"));
      if (!file_name.isEmpty()) {
        mapping_path = file_name.toStdString();
        Network::NetworkImplementation network_implementation =
            static_cast<Network::NetworkImplementation>(
                ui_.perceptronMode->currentIndex());
        controller_.ChangeImplenetation(network_implementation);

        auto mb_size = ui_.mbSize->currentText().toInt();
        controller_.SetMBSize(mb_size);
        auto epochs_count = ui_.epochs->text().toInt();
        spinner_->Start();
        timer_->start();
        chart_update_ = false;
        std::thread(&View::StartLearningTh, this, data_path, test_path,
                    mapping_path, epochs_count)
            .detach();
      }
    }
  }
}

void View::StartLearningWithCrossValidationTh(const std::string& data_path,
                                              const std::string& mapping_path,
                                              size_t k) {
  test_results_ =
      controller_.StartLearningWithCrossValidation(data_path, mapping_path, k);
  chart_update_ = true;
}

void View::StartLearningValidation() {
  std::string data_path;
  std::string mapping_path;

  QString file_name;
  file_name = QFileDialog::getOpenFileName(this, tr("Open train dataset"), ".",
                                           tr("dataset files (*.csv)"));
  if (!file_name.isEmpty()) {
    data_path = file_name.toStdString();
    file_name = QFileDialog::getOpenFileName(this, tr("Open mapping file"), ".",
                                             tr("dataset files (*.txt)"));
    if (!file_name.isEmpty()) {
      mapping_path = file_name.toStdString();
      Network::NetworkImplementation network_implementation =
          static_cast<Network::NetworkImplementation>(
              ui_.perceptronMode->currentIndex());
      controller_.ChangeImplenetation(network_implementation);
      auto mb_size = ui_.mbSize->currentText().toInt();
      controller_.SetMBSize(mb_size);
      size_t k = ui_.groupsMode->currentIndex() + 5;
      spinner_->Start();
      timer_->start();
      chart_update_ = false;
      std::thread(&View::StartLearningWithCrossValidationTh, this, data_path,
                  mapping_path, k)
          .detach();
    }
  }
}

void View::RunTest() {
  std::string data_path;
  std::string mapping_path;

  QString file_name;
  file_name = QFileDialog::getOpenFileName(this, tr("Open train dataset"), ".",
                                           tr("dataset files (*.csv)"));
  if (!file_name.isEmpty()) {
    data_path = file_name.toStdString();
    file_name = QFileDialog::getOpenFileName(this, tr("Open mapping file"), ".",
                                             tr("dataset files (*.txt)"));
    if (!file_name.isEmpty()) {
      mapping_path = file_name.toStdString();
      Network::NetworkImplementation network_implementation =
          static_cast<Network::NetworkImplementation>(
              ui_.perceptronMode->currentIndex());
      controller_.ChangeImplenetation(network_implementation);

      auto sample_part = ui_.samplePart->text().toDouble();
      auto result = controller_.RunTests(data_path, mapping_path, sample_part);
      ui_.average->setText(QString::number(result.average_accuracy, 'f', 2));
      ui_.precision->setText(QString::number(result.precision, 'f', 2));
      ui_.recall->setText(QString::number(result.recall, 'f', 2));
      ui_.f_measure->setText(QString::number(result.f_measure, 'f', 2));
      ui_.time->setText(QString::number(result.total_time, 'f', 2));
    }
  }
}

void View::LoadPerceptron() {
  auto file_name = QFileDialog::getOpenFileName(
      this, tr("Open weights from file"), ".", tr("weights files (*.txt)"));
  if (!file_name.isEmpty()) {
    controller_.LoadWeightsAndBiases(file_name.toStdString());
  }
}

void View::SavePerceptron() {
  auto file_name = QFileDialog::getSaveFileName(
      this, tr("Save weights to file"), ".", tr("weights files (*.txt)"));
  if (!file_name.isEmpty()) {
    controller_.SaveWeightsAndBiases(file_name.toStdString());
  }
}

}  // namespace s21

int main(int argc, char* argv[]) {
  QApplication a(argc, argv);
  s21::View w;
  w.show();
  return a.exec();
}
