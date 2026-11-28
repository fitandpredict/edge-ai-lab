// chip_eye: Chip is the eye, the Pi Zero is the brain (episode 34).
//   GET  /capture?size=qvga|vga|svga  -> one JPEG
//   GET  /status                      -> JSON (camera, size, RSSI, heap)
//   BOOT button (GPIO0)               -> LED flash, photo POSTed to http://PI_HOST:PI_PORT/photo
// Camera pins: ESP32-S3-EYE layout, same as episodes 03 and 33 (OV3660).
#include "esp_camera.h"
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM  4
#define SIOC_GPIO_NUM  5
#define Y9_GPIO_NUM    16
#define Y8_GPIO_NUM    17
#define Y7_GPIO_NUM    18
#define Y6_GPIO_NUM    12
#define Y5_GPIO_NUM    10
#define Y4_GPIO_NUM    8
#define Y3_GPIO_NUM    9
#define Y2_GPIO_NUM    11
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM  7
#define PCLK_GPIO_NUM  13
#include <WiFi.h>
#include <WiFiMulti.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ESPmDNS.h>
#include "secrets.h"

#define BOOT_PIN 0
#define RGB_PIN  48   // onboard WS2812 on most N16R8 dev boards (check yours)

WebServer server(80);
WiFiMulti wifi;   // joins whichever known network is strongest; reconnects to the other if one drops
static framesize_t cur = FRAMESIZE_VGA;

static void led(uint8_t r, uint8_t g, uint8_t b) { rgbLedWrite(RGB_PIN, r, g, b); }

static bool startCamera() {
  camera_config_t c = {};
  c.pin_pwdn = PWDN_GPIO_NUM;   c.pin_reset = RESET_GPIO_NUM; c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d0 = Y2_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.ledc_timer = LEDC_TIMER_0;  c.ledc_channel = LEDC_CHANNEL_0;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_SVGA;   // allocate for the largest size we serve
  c.jpeg_quality = 10;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) return false;
  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) { s->set_brightness(s, 1); s->set_saturation(s, -2); }
  s->set_vflip(s, true);
  s->set_framesize(s, cur);
  return true;
}

static void setSize(framesize_t f) {
  if (f == cur) return;
  esp_camera_sensor_get()->set_framesize(esp_camera_sensor_get(), f);
  cur = f;
  for (int i = 0; i < 2; i++) { camera_fb_t *fb = esp_camera_fb_get(); if (fb) esp_camera_fb_return(fb); }
}

static camera_fb_t *shoot() {
  camera_fb_t *old = esp_camera_fb_get();   // drop the buffered frame so the photo is from now
  if (old) esp_camera_fb_return(old);
  return esp_camera_fb_get();
}

static void handleCapture() {
  String sz = server.arg("size");
  if (sz == "qvga") setSize(FRAMESIZE_QVGA);
  else if (sz == "svga") setSize(FRAMESIZE_SVGA);
  else if (sz.length() == 0 || sz == "vga") setSize(FRAMESIZE_VGA);
  led(40, 40, 40);
  uint32_t t0 = millis();
  camera_fb_t *fb = shoot();
  led(0, 0, 0);
  if (!fb) { server.send(500, "text/plain", "capture failed"); return; }
  server.sendHeader("X-Capture-Ms", String(millis() - t0));
  server.sendHeader("X-Width", String(fb->width));
  server.sendHeader("X-Height", String(fb->height));
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

static void handleStatus() {
  sensor_t *s = esp_camera_sensor_get();
  String j = "{\"sensor_pid\":\"0x" + String(s->id.PID, HEX) + "\",\"framesize\":" + String((int)cur) +
             ",\"rssi\":" + String(WiFi.RSSI()) + ",\"heap\":" + String(ESP.getFreeHeap()) +
             ",\"psram\":" + String(ESP.getFreePsram()) + ",\"ip\":\"" + WiFi.localIP().toString() + "\"}";
  server.send(200, "application/json", j);
}

static void sendToPi() {
  led(0, 0, 60);                       // blue: taking the photo
  camera_fb_t *fb = shoot();
  if (!fb) { led(60, 0, 0); delay(500); led(0, 0, 0); return; }
  IPAddress ip = MDNS.queryHost(String(PI_HOST).substring(0, String(PI_HOST).indexOf('.')), 2000);
  String url = "http://" + (ip == IPAddress() ? String(PI_HOST) : ip.toString()) + ":" + String(PI_PORT) + "/photo";
  HTTPClient http;
  http.setTimeout(5000);
  http.begin(url);
  http.addHeader("Content-Type", "image/jpeg");
  int code = http.POST(fb->buf, fb->len);
  Serial.printf("POST %s %u bytes -> %d\n", url.c_str(), fb->len, code);
  http.end();
  esp_camera_fb_return(fb);
  led(code == 200 || code == 202 ? 0 : 60, code == 200 || code == 202 ? 60 : 0, 0);  // green ok, red fail
  delay(400);
  led(0, 0, 0);
}

void setup() {
  Serial.begin(115200);
  pinMode(BOOT_PIN, INPUT_PULLUP);
  led(30, 10, 0);
  if (!startCamera()) { for (;;) { Serial.println("ERR camera init failed"); led(60, 0, 0); delay(2000); } }
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  wifi.addAP(WIFI_SSID, WIFI_PASS);
#ifdef WIFI_SSID2
  wifi.addAP(WIFI_SSID2, WIFI_PASS2);   // backup network (on a UPS)
#endif
  while (wifi.run(10000) != WL_CONNECTED) { Serial.print('.'); }
  Serial.printf("\nWi-Fi: %s\n", WiFi.SSID().c_str());
  MDNS.begin("chip");
  MDNS.addService("http", "tcp", 80);
  server.on("/capture", handleCapture);
  server.on("/status", handleStatus);
  server.begin();
  Serial.printf("\nREADY chip_eye http://chip.local/capture (%s)\n", WiFi.localIP().toString().c_str());
  led(0, 0, 0);
}

void loop() {
  static uint32_t lastCheck = 0;
  if (millis() - lastCheck > 5000) {   // network dropped (e.g. a power cut on the extender): hop to the other one
    lastCheck = millis();
    if (WiFi.status() != WL_CONNECTED && wifi.run(8000) == WL_CONNECTED) {
      Serial.printf("Wi-Fi back: %s %s\n", WiFi.SSID().c_str(), WiFi.localIP().toString().c_str());
      MDNS.end(); MDNS.begin("chip"); MDNS.addService("http", "tcp", 80);
    }
  }
  server.handleClient();
  static uint32_t last = 0;
  if (digitalRead(BOOT_PIN) == LOW && millis() - last > 800) { last = millis(); sendToPi(); }
}
