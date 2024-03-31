#include "SettingsWidget.hpp"
#include "plugin-support.h"
#include <obs-module.h>

SettingsWidget::SettingsWidget(QWidget* parent) : QDockWidget("Test plugin", parent) {
    this->parent = parent;

    QWidget* widget = new QWidget();
    this->button->setText("Press me!");
    QHBoxLayout* layout = new QHBoxLayout();
    layout->addWidget(this->button);
    widget->setLayout(layout);

    setWidget(widget);

    setVisible(false);
    setFloating(true);
    resize(700, 700);

    connect(this->button, SIGNAL(clicked()), this, SLOT(ButtonClicked()));
}

SettingsWidget::~SettingsWidget() {}

void SettingsWidget::ButtonClicked() {
    obs_log(LOG_INFO, "aaa");
}