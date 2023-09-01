QT       += core gui charts

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17
INCLUDEPATH += model
INCLUDEPATH += controller
INCLUDEPATH += view

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    view/view.cc \
    view/draw.cc \
    view/spinner.cc \
    model/emnist.cc \
    model/layers.cc \
    model/network.cc \
    model/sigmoid.cc \
    controller/controller.cc \

HEADERS += \
    view/view.h \
    view/draw.h \
    view/spinner.h \
    model/emnist.h \
    model/layers.h \
    model/network.h \
    model/s21_matrix.h \
    model/sigmoid.h \
    controller/controller.h \

FORMS += \
    view/view.ui \
    view/draw.ui \

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
