#ifndef SETTINGS_WIDGET_H
#define SETTINGS_WIDGET_H

#include <QDockWidget>
#include <QPushButton>
#include <QHBoxLayout>
#include <QWidget>
#include <QMessageBox>

class SettingsWidget : public QDockWidget {
    Q_OBJECT
public:
    explicit SettingsWidget(QWidget* parent = nullptr);
    ~SettingsWidget();

private:
    QWidget* parent = nullptr;
    QPushButton* button = new QPushButton();

private slots:
    void ButtonClicked();
};

#endif