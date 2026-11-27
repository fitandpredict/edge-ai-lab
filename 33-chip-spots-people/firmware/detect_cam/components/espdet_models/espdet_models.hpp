#pragma once
#include <string>
#include <vector>
#include "dl_detect_base.hpp"

// Names of the espdet_pico models packed into this firmware (file names in components/espdet_models/models/)
std::vector<std::string> espdet_model_names();
// Build a detector for one of them. Same pre/post as Espressif's esp-detection template, for ESP-DL 3.3.8.
dl::detect::Detect *make_espdet(const std::string &name, float score_thr = 0.25f, float nms_thr = 0.7f);
