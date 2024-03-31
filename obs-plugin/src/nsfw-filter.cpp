#include <plugin-support.h>
#include "nsfw-filter.h"

const char* nsfw_filter_getname(void* unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("NSFWFilter");
}