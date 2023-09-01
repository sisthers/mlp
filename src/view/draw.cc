#include "draw.h"

#include "view.h"

namespace s21 {

Draw::Draw(View* view, QWidget* parent)
    : QWidget(parent), view_(view), drawing(false) {
  ui_.setupUi(this);
  connect(ui_.clear, &QPushButton::clicked, this, &Draw::Clear);
  connect(ui_.exit, &QPushButton::clicked, this, &Draw::Exit);
  ui_.drawArea->installEventFilter(this);
  image_ = new QImage(ui_.drawArea->width(), ui_.drawArea->height(),
                      QImage::Format_ARGB32_Premultiplied);
}

Draw::~Draw() {}

bool Draw::eventFilter(QObject* watched, QEvent* event) {
  if (watched == ui_.drawArea) {
    if (event->type() == QEvent::Paint) {
      QPainter painter;
      auto w = ui_.drawArea->width();
      auto h = ui_.drawArea->height();
      painter.begin(image_);
      painter.fillRect(0, 0, w, h, QBrush(Qt::black, Qt::SolidPattern));
      painter.setPen(QPen(Qt::black, Qt::SolidLine));
      painter.drawRect(0, 0, w - 1, h - 1);
      painter.setPen(QPen(Qt::white, 50, Qt::SolidLine));

      for (size_t i = 0; i < points_.size(); i++) {
        auto end = points_[i];
        if (end.x() < 0 || end.y() < 0) {
          i++;
          continue;
        }
        auto start = points_[i - 1];
        painter.drawLine(start, end);
      }
      painter.end();

      painter.begin(ui_.drawArea);
      QPixmap pix;
      pix.convertFromImage(*image_);
      painter.drawPixmap(QRect(0, 0, w, h), pix);
      painter.end();
    } else if (event->type() == QEvent::MouseButtonPress) {
      QMouseEvent* e = static_cast<QMouseEvent*>(event);
      if (e->button() == Qt::LeftButton) {
        drawing = true;
        points_.push_back(QPoint(-1, -1));
      }
    } else if (event->type() == QEvent::MouseButtonRelease) {
      QMouseEvent* e = static_cast<QMouseEvent*>(event);
      if (e->button() == Qt::LeftButton) {
        drawing = false;
      }
    } else if (event->type() == QEvent::MouseMove) {
      QMouseEvent* e = static_cast<QMouseEvent*>(event);
      if (drawing) {
        points_.push_back(e->pos());
        update();
      }
    }
  }
  return true;
}

void Draw::Clear() {
  points_.clear();
  update();
}

void Draw::Exit() {
  view_->image_ = *image_;
  view_->update();
  close();
  Clear();
}

void Draw::closeEvent(QCloseEvent* event) {
  Q_UNUSED(event);
  Exit();
}

}  // namespace s21
