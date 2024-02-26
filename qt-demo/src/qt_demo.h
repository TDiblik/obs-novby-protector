#pragma once
#include "ui_qt_demo.h"
#include <QMainWindow>

class qt_demo : public QMainWindow {
    Q_OBJECT
    
public:
    qt_demo(QWidget* parent = nullptr);
    ~qt_demo();

private:
    Ui_qt_demo* ui;
};