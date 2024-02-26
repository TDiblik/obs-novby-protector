#include "qt_demo.h"

#include <QApplication>
#pragma comment(lib, "user32.lib")

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    qt_demo w;
    w.show();
    return a.exec();
}