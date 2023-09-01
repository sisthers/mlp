#include "spinner.h"

Spinner::Spinner(QWidget *parent, bool center_on_parent,
                 bool disable_parent_when_spinning)
    : QWidget(parent),
      center_on_parent_(center_on_parent),
      disable_parent_when_spinning_(disable_parent_when_spinning) {
  Initialize();
}

Spinner::Spinner(Qt::WindowModality modality, QWidget *parent,
                 bool center_on_parent, bool disable_parent_when_spinning)
    : QWidget(parent, Qt::Dialog | Qt::FramelessWindowHint),
      center_on_parent_(center_on_parent),
      disable_parent_when_spinning_(disable_parent_when_spinning) {
  Initialize();
  setWindowModality(modality);
  setAttribute(Qt::WA_TranslucentBackground);
}

void Spinner::Initialize() {
  color_ = Qt::black;
  roundness_ = 100.0;
  minimum_trail_opacity_ = 3.14159265358979323846;
  trail_fade_percentage_ = 80.0;
  revolutions_per_second_ = 1.57079632679489661923;
  number_of_lines_ = 20;
  line_length_ = 10;
  line_width_ = 2;
  inner_radius_ = 10;
  current_counter_ = 0;
  is_spinning_ = false;

  timer_ = new QTimer(this);
  connect(timer_, &QTimer::timeout, this, &Spinner::Rotate);
  UpdateSize();
  UpdateTimer();
  hide();
}

void Spinner::paintEvent(QPaintEvent *) {
  UpdatePosition();
  QPainter painter(this);
  painter.fillRect(this->rect(), Qt::transparent);
  painter.setRenderHint(QPainter::Antialiasing, true);

  if (current_counter_ >= number_of_lines_) {
    current_counter_ = 0;
  }

  painter.setPen(Qt::NoPen);
  for (int i = 0; i < number_of_lines_; ++i) {
    painter.save();
    painter.translate(inner_radius_ + line_length_,
                      inner_radius_ + line_length_);
    qreal rotateAngle =
        static_cast<qreal>(360 * i) / static_cast<qreal>(number_of_lines_);
    painter.rotate(rotateAngle);
    painter.translate(inner_radius_, 0);
    int distance =
        LineCountDistanceFromPrimary(i, current_counter_, number_of_lines_);
    QColor color =
        CurrentLineColor(distance, number_of_lines_, trail_fade_percentage_,
                         minimum_trail_opacity_, color_);
    painter.setBrush(color);
    painter.drawRoundedRect(
        QRect(0, -line_width_ / 2, line_length_, line_width_), roundness_,
        roundness_, Qt::RelativeSize);
    painter.restore();
  }
}

void Spinner::Start() {
  UpdatePosition();
  is_spinning_ = true;
  show();

  if (parentWidget() && disable_parent_when_spinning_) {
    parentWidget()->setEnabled(false);
  }

  if (!timer_->isActive()) {
    timer_->start();
    current_counter_ = 0;
  }
}

void Spinner::Stop() {
  is_spinning_ = false;
  hide();

  if (parentWidget() && disable_parent_when_spinning_) {
    parentWidget()->setEnabled(true);
  }

  if (timer_->isActive()) {
    timer_->stop();
    current_counter_ = 0;
  }
}

void Spinner::SetNumberOfLines(int lines) {
  number_of_lines_ = lines;
  current_counter_ = 0;
  UpdateTimer();
}

void Spinner::SetLineLength(int length) {
  line_length_ = length;
  UpdateSize();
}

void Spinner::SetLineWidth(int width) {
  line_width_ = width;
  UpdateSize();
}

void Spinner::SetInnerRadius(int radius) {
  inner_radius_ = radius;
  UpdateSize();
}

QColor Spinner::Color() { return color_; }

qreal Spinner::Roundness() { return roundness_; }

qreal Spinner::MinimumTrailOpacity() { return minimum_trail_opacity_; }

qreal Spinner::TrailFadePercentage() { return trail_fade_percentage_; }

qreal Spinner::RevolutionsPersSecond() { return revolutions_per_second_; }

int Spinner::NumberOfLines() { return number_of_lines_; }

int Spinner::LineLength() { return line_length_; }

int Spinner::LineWidth() { return line_width_; }

int Spinner::InnerRadius() { return inner_radius_; }

bool Spinner::IsSpinning() const { return is_spinning_; }

void Spinner::SetRoundness(qreal roundness) {
  roundness_ = std::max(0.0, std::min(100.0, roundness));
}

void Spinner::SetColor(QColor color) { color_ = color; }

void Spinner::SetRevolutionsPerSecond(qreal revolutions_per_second) {
  revolutions_per_second_ = revolutions_per_second;
  UpdateTimer();
}

void Spinner::SetTrailFadePercentage(qreal trail) {
  trail_fade_percentage_ = trail;
}

void Spinner::SetMinimumTrailOpacity(qreal minimum_trail_opacity) {
  minimum_trail_opacity_ = minimum_trail_opacity;
}

void Spinner::Rotate() {
  ++current_counter_;
  if (current_counter_ >= number_of_lines_) {
    current_counter_ = 0;
  }
  update();
}

void Spinner::UpdateSize() {
  int size = (inner_radius_ + line_length_) * 2;
  setFixedSize(size, size);
}

void Spinner::UpdateTimer() {
  timer_->setInterval(1000 / (number_of_lines_ * revolutions_per_second_));
}

void Spinner::UpdatePosition() {
  if (parentWidget() && center_on_parent_) {
    move(parentWidget()->width() / 2 - width() / 2,
         parentWidget()->height() / 2 - height() / 2);
  }
}

int Spinner::LineCountDistanceFromPrimary(int current, int primary,
                                          int total_nr_of_lines) {
  int distance = primary - current;
  if (distance < 0) {
    distance += total_nr_of_lines;
  }
  return distance;
}

QColor Spinner::CurrentLineColor(int count_distance, int total_nr_of_lines,
                                 qreal trail_fade_perc, qreal minOpacity,
                                 QColor color) {
  if (count_distance == 0) {
    return color;
  }
  const qreal min_alpha_f = minOpacity / 100.0;
  int distance_threshold =
      static_cast<int>(ceil((total_nr_of_lines - 1) * trail_fade_perc / 100.0));
  if (count_distance > distance_threshold) {
    color.setAlphaF(min_alpha_f);
  } else {
    qreal alpha_diff = color.alphaF() - min_alpha_f;
    qreal gradient = alpha_diff / static_cast<qreal>(distance_threshold + 1);
    qreal result_alpha = color.alphaF() - gradient * count_distance;

    result_alpha = std::min(1.0, std::max(0.0, result_alpha));
    color.setAlphaF(result_alpha);
  }
  return color;
}
