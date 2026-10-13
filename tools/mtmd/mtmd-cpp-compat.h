#pragma once

#include "../../ggml/src/ggml-cpp-compat.h"

#ifdef __k1om__
#include <experimental/string_view>
using mtmd_string_view = std::experimental::string_view;
#else
#include <string_view>
using mtmd_string_view = std::string_view;
#endif
