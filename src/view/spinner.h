#ifndef CPP7_MLP_VIEW_SPINNER_H_
#define CPP7_MLP_VIEW_SPINNER_H_

#include <QColor>
#include <QPainter>
#include <QTimer>
#include <QWidget>
#include <algorithm>
#include <cmath>

class Spinner : public QWidget {
  Q_OBJECT
 public:
  Spinner(QWidget *parent = 0, bool center_on_parent = true,
          bool disable_parent_when_spinning = true);

  Spinner(Qt::WindowModality modality, QWidget *parent = 0,
          bool center_on_parent = true,
          bool disable_parent_when_spinning = true);

 public slots:
  void Start();
  void Stop();

 public:
  void SetColor(QColor color);
  void SetRoundness(qreal roundness);
  void SetMinimumTrailOpacity(qreal minimum_trail_opacity);
  void SetTrailFadePercentage(qreal trail);
  void SetRevolutionsPerSecond(qreal revolutions_per_second);
  void SetNumberOfLines(int lines);
  void SetLineLength(int length);
  void SetLineWidth(int width);
  void SetInnerRadius(int radius);

  QColor Color();
  qreal Roundness();
  qreal MinimumTrailOpacity();
  qreal TrailFadePercentage();
  qreal RevolutionsPersSecond();
  int NumberOfLines();
  int LineLength();
  int LineWidth();
  int InnerRadius();

  bool IsSpinning() const;

 private slots:
  void Rotate();

 protected:
  void paintEvent(QPaintEvent *paintEvent);

 private:
  static int LineCountDistanceFromPrimary(int current, int primary,
                                          int total_nr_of_lines);
  static QColor CurrentLineColor(int distance, int total_nr_of_lines,
                                 qreal trailFadePerc, qreal min_opacity,
                                 QColor color);

  void Initialize();
  void UpdateSize();
  void UpdateTimer();
  void UpdatePosition();

 private:
  QColor color_;
  qreal roundness_;
  qreal minimum_trail_opacity_;
  qreal trail_fade_percentage_;
  qreal revolutions_per_second_;
  int number_of_lines_;
  int line_length_;
  int line_width_;
  int inner_radius_;

  Spinner(const Spinner &);
  Spinner &operator=(const Spinner &);

  QTimer *timer_;
  bool center_on_parent_;
  bool disable_parent_when_spinning_;
  int current_counter_;
  bool is_spinning_;
};

#endif  // CPP7_MLP_VIEW_SPINNER_H_