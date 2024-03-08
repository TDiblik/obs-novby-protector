#ifndef TEST_H
#define TEST_H

#include <QDockWidget>
#include <QPushButton>
#include <QHBoxLayout>
#include <QWidget>
#include <QMessageBox>

class SettingsWidget : public QDockWidget {
    QOBJECT_H
public:
    explicit SettingsWidget(QWidget* parent = nullptr);
    ~SettingsWidget();

private:
    void buttonClicked();
    QWidget* parent = nullptr;
    QPushButton* button = new QPushButton();

private slots:
    void ButtonClicked();
};

#endif