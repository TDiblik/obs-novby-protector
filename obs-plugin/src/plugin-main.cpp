#include <obs-module.h>

#include "SettingsWidget.hpp"
#include <QWidget>
#include <obs-frontend-api.h>

#include <plugin-support.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

extern struct obs_source_info nsfw_filter_info;
bool obs_module_load(void)
{
	QWidget* main_window = (QWidget*)obs_frontend_get_main_window();
	SettingsWidget* test_widget = new SettingsWidget(main_window);
	obs_frontend_add_dock_by_id(PLUGIN_NAME, "Real-time NSFW filtering", test_widget);
	obs_register_source(&nsfw_filter_info);
	obs_log(LOG_INFO, "plugin loaded successfully (version %s)",
		PLUGIN_VERSION);
	return true;
}

void obs_module_unload(void)
{
	obs_log(LOG_INFO, "plugin unloaded");
}
