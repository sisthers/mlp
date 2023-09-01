#ifndef CPP7_MLP_VIEW_DRAW_H_
#define CPP7_MLP_VIEW_DRAW_H_

#include <QImage>
#include <QMouseEvent>
#include <QPainter>
#include <QWidget>
#include <vector>

#include "ui_draw.h"

namespace s21 {

class View;

class Draw : public QWidget {
  Q_OBJECT

 public:
  Draw(View* view = nullptr, QWidget* parent = nullptr);
  ~Draw();

 private slots:
  void Clear();
  void Exit();

 private:
  Ui::DrawClass ui_;
  View* view_;
  bool drawing;
  std::vector<QPoint> points_;
  QImage* image_;

  bool eventFilter(QObject* watched, QEvent* event);
  void closeEvent(QCloseEvent* event);
};

}  // namespace s21

#endif  // CPP7_MLP_VIEW_DRAW_H_