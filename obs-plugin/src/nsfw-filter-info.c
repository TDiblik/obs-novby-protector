#include "nsfw-filter.h"

struct obs_source_info nsfw_filter_info = {
	.id = "nsfw_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = nsfw_filter_getname,
	.create = nsfw_filter_create,
	.destroy = nsfw_filter_destroy,
	.get_defaults = nsfw_filter_defaults,
	.get_properties = nsfw_filter_properties,
	.update = nsfw_filter_update,
	.activate = nsfw_filter_activate,
	.deactivate = nsfw_filter_deactivate,
	.video_render = nsfw_filter_video_render,
};
