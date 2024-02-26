#include "qt_demo.h"

qt_demo::qt_demo(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui_qt_demo)
{
    ui->setupUi(this);
}

qt_demo::~qt_demo()
{
    delete ui; 
}