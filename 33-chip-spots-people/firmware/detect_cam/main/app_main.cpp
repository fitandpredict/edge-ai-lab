// Rung 0b+: YOLO11n (ESP-DL 3.3.8) on Chip's live OV3660 camera, every stage timed.
//
// Serial protocol (USB-Serial/JTAG, /dev/cu.usbmodem101), read by tools/chip_view.py:
//   text  RUNG ... one line per frame with timings and boxes
//   bin   "\n@@JPG <len> <frame>\n" + <len> JPEG bytes: the frame the boxes belong to (when preview is on)
// Keys from the laptop: '0' = YOLO11n 640, '1' = YOLO11n 320, '2'.. = our espdet_pico models (in the order
// MODELS lists them at boot), 'p' = preview on/off, 'i' = inject mode on/off.
// Inject mode (tools/feed_clips.py): the camera is idle; the laptop sends 'J' + 4-byte little-endian length + a
// JPEG, Chip decodes it and runs the same timed detector on it, and prints the same RUNG line with src=inject.
#include <cstdio>
#include <cstring>

#include "coco_detect.hpp"
#include "dl_image_jpeg.hpp"
#include "espdet_models.hpp"
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "img_converters.h"

static const char *TAG = "detect_cam";

// ESP32-S3-EYE layout, probed on this board in episode 03 (code/README.md)
static camera_config_t cam_cfg()
{
    camera_config_t c = {};
    c.pin_pwdn = -1;
    c.pin_reset = -1;
    c.pin_xclk = 15;
    c.pin_sccb_sda = 4;
    c.pin_sccb_scl = 5;
    c.pin_d7 = 16;
    c.pin_d6 = 17;
    c.pin_d5 = 18;
    c.pin_d4 = 12;
    c.pin_d3 = 10;
    c.pin_d2 = 8;
    c.pin_d1 = 9;
    c.pin_d0 = 11;
    c.pin_vsync = 6;
    c.pin_href = 7;
    c.pin_pclk = 13;
    c.xclk_freq_hz = 20000000;
    c.ledc_timer = LEDC_TIMER_0;
    c.ledc_channel = LEDC_CHANNEL_0;
    c.pixel_format = PIXFORMAT_RGB565;
    c.frame_size = FRAMESIZE_VGA;  // 640x480: the 640 model sees the same pixels
    c.jpeg_quality = 12;
    c.fb_count = 3;  // 2-core mode: one waiting, one being detected, one the camera is filling
    c.fb_location = CAMERA_FB_IN_PSRAM;
    c.grab_mode = CAMERA_GRAB_LATEST;  // always the newest frame, never a stale queued one
    return c;
}

static bool g_qvga = false;
static esp_err_t start_camera()
{
    camera_config_t c = cam_cfg();
    if (g_qvga) c.frame_size = FRAMESIZE_QVGA;
    esp_err_t e = esp_camera_init(&c);
    sensor_t *s = esp_camera_sensor_get();
    if (e == ESP_OK && s && s->id.PID == OV3660_PID) {  // same as episode 03: without these the frames come out green
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
    }
    return e;
}

// id 0/1: Espressif's YOLO11n (COCO, 80 classes); id >= 2: our espdet_pico models
static std::vector<std::string> g_espdet;
static dl::detect::Detect *make_detector(int id)
{
    // Built directly (not via the COCODetect wrapper) so every detector is a DetectImpl we can time stage by stage
    if (id == 0) return new coco_detect::Yolo11n("coco_detect_yolo11n_s8_v1.espdl", 0.25, 0.7);
    if (id == 1) return new coco_detect::Yolo11n("coco_detect_yolo11n_320_s8_v1.espdl", 0.25, 0.7);
    return make_espdet(g_espdet[id - 2]);
}

// ESP-DL's own pre/model/post logging is compiled out (dl_define.hpp hard-sets DL_LOG_INFER_LATENCY 0), so this
// repeats DetectImpl::run (ESP-DL 3.3.8, vision/detect/dl_detect_base.cpp) with a timer around each stage.
struct Stages {
    double pre_ms, model_ms, post_ms;
};
struct TimedDetect : dl::detect::DetectImpl {
    static std::list<dl::detect::result_t> &timed_run(dl::detect::Detect *d, const dl::image::img_t &img, Stages &st)
    {
        auto *t = static_cast<TimedDetect *>(static_cast<dl::detect::DetectImpl *>(d));
        int64_t a = esp_timer_get_time();
        t->m_image_preprocessor->preprocess(img);
        int64_t b = esp_timer_get_time();
        t->m_model->run();
        int64_t c = esp_timer_get_time();
        t->m_postprocessor->clear_result();
        t->m_postprocessor->postprocess();
        auto &res = t->m_postprocessor->get_result(img.width, img.height);
        int64_t e = esp_timer_get_time();
        st = {(b - a) / 1000.0, (c - b) / 1000.0, (e - c) / 1000.0};
        return res;
    }
};
static int n_models() { return 2 + (int)g_espdet.size(); }

// Read exactly n bytes from the laptop (blocks up to timeout_ms between chunks)
static bool read_exact(uint8_t *p, size_t n, int timeout_ms)
{
    while (n) {
        int r = usb_serial_jtag_read_bytes(p, n > 4096 ? 4096 : n, pdMS_TO_TICKS(timeout_ms));
        if (r <= 0) return false;
        p += r;
        n -= r;
    }
    return true;
}

static void print_rung(int model_id, uint32_t frame, const char *src, int w, int h, double capture_ms,
                       double detect_ms, const Stages &st, std::list<dl::detect::result_t> &results)
{
    char boxes[512];
    int off = 0, people = 0;
    for (const auto &r : results) {
        if (r.category == 0) people++;  // COCO class 0 and our one-class models: person (or cat)
        if (off < (int)sizeof(boxes) - 48)
            off += snprintf(boxes + off, sizeof(boxes) - off, "%s%d:%.2f:%d,%d,%d,%d", off ? ";" : "", r.category,
                            r.score, r.box[0], r.box[1], r.box[2], r.box[3]);
    }
    boxes[off] = 0;
    printf("RUNG model=%d frame=%lu src=%s img=%dx%d capture_ms=%.1f detect_ms=%.1f pre_ms=%.1f model_ms=%.1f "
           "post_ms=%.1f people=%d boxes=%s\n",
           model_id, (unsigned long)frame, src, w, h, capture_ms, detect_ms, st.pre_ms, st.model_ms, st.post_ms,
           people, boxes);
}

static void write_bin(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    while (n) {
        int w = usb_serial_jtag_write_bytes(b, n > 512 ? 512 : n, pdMS_TO_TICKS(200));
        if (w <= 0) continue;
        b += w;
        n -= w;
    }
}

// 2-core mode ('m' toggles): the capture task on core 0 grabs the next frame while core 1 detects
static QueueHandle_t g_frames;
static volatile bool g_two_core = false;
static void capture_task(void *)
{
    for (;;) {
        if (!g_two_core) {
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb && xQueueSend(g_frames, &fb, pdMS_TO_TICKS(30000)) != pdTRUE) esp_camera_fb_return(fb);
    }
}
static void detect_loop(void *);

extern "C" void app_main(void)
{
    usb_serial_jtag_driver_config_t ucfg = {.tx_buffer_size = 16384, .rx_buffer_size = 16384};
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&ucfg));
    usb_serial_jtag_vfs_use_driver();

    camera_config_t cfg = cam_cfg();
    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed: %s", esp_err_to_name(err));
        return;
    }
    sensor_t *s = esp_camera_sensor_get();
    ESP_LOGI(TAG, "camera PID 0x%04x", s->id.PID);
    if (s->id.PID == OV3660_PID) {  // same as episode 03: without these the OV3660 frames come out green
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
    }

    g_espdet = espdet_model_names();
    printf("MODELS 0=coco_detect_yolo11n_s8_v1 1=coco_detect_yolo11n_320_s8_v1");
    for (size_t i = 0; i < g_espdet.size(); i++) printf(" %d=%s", (int)i + 2, g_espdet[i].c_str());
    printf("\n");

    // Core 1 runs the detector; core 0 keeps the camera driver (CONFIG_CAMERA_CORE0) and, in 2-core mode,
    // a capture task that has the next frame waiting before the detector asks for it.
    g_frames = xQueueCreate(1, sizeof(camera_fb_t *));
    xTaskCreatePinnedToCore(capture_task, "capture", 4096, nullptr, 5, nullptr, 0);
    xTaskCreatePinnedToCore(detect_loop, "detect", 32 * 1024, nullptr, 5, nullptr, 1);
}

static void detect_loop(void *)
{
    int model_id = 0;
    bool preview = true, inject = false, keep_cam = false;
    uint8_t *jbuf = (uint8_t *)heap_caps_malloc(512 * 1024, MALLOC_CAP_SPIRAM);
    int64_t tl = esp_timer_get_time();
    dl::detect::Detect *detect = make_detector(model_id);
    printf("MODEL id=%d load_ms=%.1f psram_free=%u\n", model_id, (esp_timer_get_time() - tl) / 1000.0,
           heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "free psram %u B, internal %u B", heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    for (uint32_t frame = 0;; frame++) {
        uint8_t key;
        bool got_jpeg = false;
        uint32_t jlen = 0;
        while (!got_jpeg && usb_serial_jtag_read_bytes(&key, 1, inject ? pdMS_TO_TICKS(50) : 0) == 1) {
            if (key == 'J' && inject) {
                if (read_exact((uint8_t *)&jlen, 4, 1000) && jlen > 0 && jlen <= 512 * 1024 &&
                    read_exact(jbuf, jlen, 2000))
                    got_jpeg = true;
                else
                    printf("INJECT error len=%lu\n", (unsigned long)jlen);
                continue;
            }
            int want = key - '0';
            if (want >= 0 && want < n_models() && want != model_id) {
                delete detect;
                model_id = want;
                int64_t t = esp_timer_get_time();
                detect = make_detector(model_id);
                printf("MODEL id=%d load_ms=%.1f psram_free=%u\n", model_id, (esp_timer_get_time() - t) / 1000.0,
                       heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
            } else if (key == 'p') {
                preview = !preview;
            } else if (key == 'i') {
                inject = !inject;
                // camera off while injecting: an idle camera keeps DMA-writing frames into PSRAM, the same memory
                // the network streams its weights from ('k' keeps it on, to measure that)
                if (inject && !keep_cam) esp_camera_deinit();
                if (!inject && !keep_cam) {
                    start_camera();
                }
                printf("INJECT %s camera=%s\n", inject ? "on" : "off", inject && !keep_cam ? "off" : "on");
            } else if (key == 'q') {  // camera frame size VGA 640x480 <-> QVGA 320x240 (the driver needs a restart)
                g_qvga = !g_qvga;
                bool two = g_two_core;
                g_two_core = false;
                camera_fb_t *old;
                for (int k = 0; k < 3; k++) {  // the capture task may be mid-send with one more frame
                    vTaskDelay(pdMS_TO_TICKS(100));
                    while (xQueueReceive(g_frames, &old, 0) == pdTRUE) esp_camera_fb_return(old);
                }
                esp_camera_deinit();
                start_camera();
                g_two_core = two;
                printf("FRAMESIZE %s\n", g_qvga ? "320x240" : "640x480");
            } else if (key == 'm') {
                g_two_core = !g_two_core;
                if (!g_two_core) {  // hand back a frame the capture task left waiting
                    camera_fb_t *old;
                    vTaskDelay(pdMS_TO_TICKS(50));
                    while (xQueueReceive(g_frames, &old, 0) == pdTRUE) esp_camera_fb_return(old);
                }
                printf("CORES %d\n", g_two_core ? 2 : 1);
            } else if (key == 'k') {
                keep_cam = !keep_cam;
            }
        }
        if (inject) {
            if (!got_jpeg) {
                vTaskDelay(1);
                frame--;
                continue;
            }
            dl::image::jpeg_img_t jpeg = {.data = jbuf, .data_len = jlen};
            int64_t a = esp_timer_get_time();
            dl::image::img_t img = dl::image::sw_decode_jpeg(jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
            int64_t b = esp_timer_get_time();
            Stages st;
            auto &results = TimedDetect::timed_run(detect, img, st);
            int64_t c = esp_timer_get_time();
            print_rung(model_id, frame, "inject", img.width, img.height, (b - a) / 1000.0, (c - b) / 1000.0, st,
                       results);
            heap_caps_free(img.data);
            fflush(stdout);
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        int64_t t0 = esp_timer_get_time();
        camera_fb_t *fb = nullptr;
        if (g_two_core)
            xQueueReceive(g_frames, &fb, pdMS_TO_TICKS(1000));
        else
            fb = esp_camera_fb_get();
        int64_t t1 = esp_timer_get_time();
        if (!fb) {
            ESP_LOGW(TAG, "no frame");
            continue;
        }
        dl::image::img_t img = {.data = fb->buf, .width = (uint16_t)fb->width, .height = (uint16_t)fb->height,
                                .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE};
        Stages st;
        auto &results = TimedDetect::timed_run(detect, img, st);
        int64_t t2 = esp_timer_get_time();

        static int64_t last_ts = 0;  // the camera's own pace: time between the frames it hands over
        int64_t ts = (int64_t)fb->timestamp.tv_sec * 1000000 + fb->timestamp.tv_usec;
        printf("CAMGAP ms=%.1f\n", last_ts ? (ts - last_ts) / 1000.0 : 0.0);
        last_ts = ts;
        print_rung(model_id, frame, g_two_core ? "camera2" : "camera", fb->width, fb->height, (t1 - t0) / 1000.0, (t2 - t1) / 1000.0, st,
                   results);

        if (preview) {  // the frame the boxes were computed on, so the overlay shows where you *were*
            uint8_t *jpg = nullptr;
            size_t len = 0;
            int64_t t3 = esp_timer_get_time();
            if (frame2jpg(fb, 70, &jpg, &len)) {
                printf("JPGTIME frame=%lu encode_ms=%.1f\n", (unsigned long)frame, (esp_timer_get_time() - t3) / 1000.0);
                fflush(stdout);
                char hdr[48];
                int h = snprintf(hdr, sizeof(hdr), "\n@@JPG %u %lu\n", (unsigned)len, (unsigned long)frame);
                write_bin(hdr, h);
                write_bin(jpg, len);
                free(jpg);
            }
        }
        esp_camera_fb_return(fb);
        fflush(stdout);
        if (!g_two_core) vTaskDelay(pdMS_TO_TICKS(10));  // 1-core mode as measured for the ladder
    }
}
