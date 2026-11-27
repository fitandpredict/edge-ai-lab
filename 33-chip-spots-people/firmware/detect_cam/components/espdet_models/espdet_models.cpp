#include "espdet_models.hpp"
#include "dl_detect_espdet_postprocessor.hpp"
#include "esp_heap_caps.h"

extern const uint8_t espdet_models_espdl[] asm("_binary_espdet_models_espdl_start");

namespace {
class ESPDet : public dl::detect::DetectImpl {
public:
    ESPDet(const char *model_name, float score_thr, float nms_thr)
    {
        m_model = new dl::Model((const char *)espdet_models_espdl, model_name, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0,
                                dl::MEMORY_MANAGER_GREEDY, nullptr, true);
        m_model->minimize();
        m_image_preprocessor = new dl::image::ImagePreprocessor(m_model, {0, 0, 0}, {255, 255, 255});
        m_image_preprocessor->enable_letterbox({114, 114, 114});
        m_postprocessor = new dl::detect::ESPDetPostProcessor(m_model, m_image_preprocessor, score_thr, nms_thr, 10,
                                                              {{8, 8, 4, 4}, {16, 16, 8, 8}, {32, 32, 16, 16}});
    }
};
}  // namespace

std::vector<std::string> espdet_model_names()
{
    std::vector<std::string> out;
    std::string s = ESPDET_MODEL_NAMES, cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

dl::detect::Detect *make_espdet(const std::string &name, float score_thr, float nms_thr)
{
    return new ESPDet(name.c_str(), score_thr, nms_thr);
}
