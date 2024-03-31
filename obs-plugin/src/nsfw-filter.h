#ifndef NSFW_FILTER_H
#define NSFW_FILTER_H

#include <obs-module.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *nsfw_filter_getname(void *unused);
void *nsfw_filter_create(obs_data_t *settings, obs_source_t *source);
void nsfw_filter_destroy(void *data);
void nsfw_filter_defaults(obs_data_t *settings);
obs_properties_t *nsfw_filter_properties(void *data);
void nsfw_filter_update(void *data, obs_data_t *settings);
void nsfw_filter_activate(void *data);
void nsfw_filter_deactivate(void *data);
void nsfw_filter_video_tick(void *data, float seconds);
void nsfw_filter_video_render(void *data, gs_effect_t *_effect);

#ifdef __cplusplus
}
#endif

#endif
