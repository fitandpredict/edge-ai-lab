// Chip is the maze rover's slow brain: a 128-unit LSTM + small head, once per cell.
// The MuJoCo/cell sim on the Mac sends what the six lasers, encoders and gyro say; Chip answers
// forward / left / right / back. The LSTM memory (h, c) lives on Chip between cells.
//
// Protocol (binary, little endian):
//   Mac -> Chip   AA 55 + 20 x float32 observations
//   Chip -> Mac   55 AA + 4 x float32 logits + uint8 action + uint32 inference microseconds
// Text commands (one byte):
//   'R'       new maze: clear the LSTM memory
//   '0'..'4'  pick how the brain runs (see MODE_NAME), answered with a JSON line
//   'B'       benchmark every mode (200 runs each), JSON lines
//   'I'       info: params, sizes, free RAM, current mode
//
// policy_maze.h comes from sim/export_brain.py.

#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include "esp_dsp.h"
#include "policy_maze.h"

#define N_IN 20
#define N_H 128
#define N_ACT 4
#define MAX_W 512

enum Mode { F32_FLASH = 0, F32_PSRAM, F32_PSRAM_SIMD, F32_SRAM_SIMD, I8_SRAM, N_MODES };
static const char* MODE_NAME[N_MODES] = {
    "float32, weights read from flash, plain C",
    "float32, weights in PSRAM, plain C",
    "float32, weights in PSRAM, SIMD dot product",
    "float32, weights in internal RAM, SIMD dot product",
    "int8 weights in internal RAM, plain C",
};

static const float* W_psram[BRAIN_N_LAYERS];
static const float* W_sram[BRAIN_N_LAYERS];
static const int8_t* W_i8[BRAIN_N_LAYERS];
static bool sram_f32_ok = true;
static Mode mode = F32_PSRAM_SIMD;

static float h[N_H] __attribute__((aligned(16)));
static float c[N_H] __attribute__((aligned(16)));
static float z[N_IN + N_H] __attribute__((aligned(16)));   // [x ; h]
static float g[MAX_W] __attribute__((aligned(16)));
static float hid[MAX_W] __attribute__((aligned(16)));
static int8_t qbuf[MAX_W] __attribute__((aligned(16)));

static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

static void dense_f32(const float* W, const float* b, const float* x, float* y, int nin, int nout, bool simd) {
  for (int o = 0; o < nout; o++) {
    const float* w = W + o * nin;
    float acc;
    if (simd) {
      dsps_dotprod_f32(w, x, &acc, nin);
    } else {
      acc = 0;
      for (int i = 0; i < nin; i++) acc += w[i] * x[i];
    }
    y[o] = acc + b[o];
  }
}

static void dense_i8(const int8_t* W, const float* s, const float* b, const float* x, float* y, int nin, int nout) {
  float amax = 1e-8f;
  for (int i = 0; i < nin; i++) amax = fmaxf(amax, fabsf(x[i]));
  const float xs = amax / 127.0f, inv = 127.0f / amax;
  for (int i = 0; i < nin; i++) qbuf[i] = (int8_t)lrintf(x[i] * inv);
  for (int o = 0; o < nout; o++) {
    const int8_t* w = W + o * nin;
    int32_t acc = 0;
    for (int i = 0; i < nin; i++) acc += (int32_t)w[i] * qbuf[i];
    y[o] = acc * s[o] * xs + b[o];
  }
}

static void layer(int l, const float* x, float* y) {
  int nin = BRAIN_IN[l], nout = BRAIN_OUT[l];
  switch (mode) {
    case F32_FLASH:      dense_f32(BRAIN_F32[l], BRAIN_B[l], x, y, nin, nout, false); break;
    case F32_PSRAM:      dense_f32(W_psram[l], BRAIN_B[l], x, y, nin, nout, false); break;
    case F32_PSRAM_SIMD: dense_f32(W_psram[l], BRAIN_B[l], x, y, nin, nout, true); break;
    case F32_SRAM_SIMD:  dense_f32(W_sram[l], BRAIN_B[l], x, y, nin, nout, true); break;
    default:             dense_i8(W_i8[l], BRAIN_S[l], BRAIN_B[l], x, y, nin, nout); break;
  }
}

static void reset_memory() {
  memset(h, 0, sizeof h);
  memset(c, 0, sizeof c);
}

// one decision: LSTM step (gate order i, f, g, o as in PyTorch), tanh layer, logits
static uint32_t think(const float* obs, float* logits, uint8_t* action) {
  int64_t t0 = esp_timer_get_time();
  memcpy(z, obs, N_IN * 4);
  memcpy(z + N_IN, h, N_H * 4);
  layer(0, z, g);
  for (int k = 0; k < N_H; k++) {
    float i = sigm(g[k]), f = sigm(g[N_H + k]), gg = tanhf(g[2 * N_H + k]), o = sigm(g[3 * N_H + k]);
    c[k] = f * c[k] + i * gg;
    h[k] = o * tanhf(c[k]);
  }
  layer(1, h, hid);
  for (int k = 0; k < BRAIN_OUT[1]; k++) hid[k] = tanhf(hid[k]);
  layer(2, hid, logits);
  uint8_t best = 0;
  for (int k = 1; k < N_ACT; k++) if (logits[k] > logits[best]) best = k;
  *action = best;
  return (uint32_t)(esp_timer_get_time() - t0);
}

static void info() {
  Serial.printf("{\"params_on_chip\":%d,\"params_torch\":%d,\"float32_kib\":%.1f,\"int8_kib\":%.1f,\"mode\":%d,"
                "\"mode_name\":\"%s\",\"sram_f32_fits\":%s,\"free_internal\":%u,\"free_psram\":%u}\n",
                BRAIN_PARAMS, BRAIN_PARAMS_TORCH, BRAIN_PARAMS * 4 / 1024.0, BRAIN_PARAMS / 1024.0, mode,
                MODE_NAME[mode], sram_f32_ok ? "true" : "false",
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

static void bench() {
  float obs[N_IN], logits[N_ACT];
  uint8_t a;
  for (int i = 0; i < N_IN; i++) obs[i] = 0.1f * sinf(i);
  Mode keep = mode;
  for (int m = 0; m < N_MODES; m++) {
    if (m == F32_SRAM_SIMD && !sram_f32_ok) {
      Serial.printf("{\"mode\":%d,\"name\":\"%s\",\"skipped\":\"does not fit in internal RAM\"}\n", m, MODE_NAME[m]);
      continue;
    }
    mode = (Mode)m;
    reset_memory();
    think(obs, logits, &a);
    uint32_t best = UINT32_MAX, sum = 0;
    for (int k = 0; k < 200; k++) {
      uint32_t us = think(obs, logits, &a);
      sum += us;
      if (us < best) best = us;
    }
    Serial.printf("{\"mode\":%d,\"name\":\"%s\",\"mean_us\":%.1f,\"best_us\":%u}\n", m, MODE_NAME[m], sum / 200.0, best);
  }
  mode = keep;
  reset_memory();
}

void setup() {
  Serial.setRxBufferSize(1024);
  Serial.begin(921600);
  for (int l = 0; l < BRAIN_N_LAYERS; l++) {
    size_t n = (size_t)BRAIN_IN[l] * BRAIN_OUT[l];
    float* p = (float*)heap_caps_aligned_alloc(16, n * 4, MALLOC_CAP_SPIRAM);
    memcpy(p, BRAIN_F32[l], n * 4);
    W_psram[l] = p;
    int8_t* q = (int8_t*)heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    memcpy(q, BRAIN_I8[l], n);
    W_i8[l] = q;
  }
  for (int l = 0; l < BRAIN_N_LAYERS && sram_f32_ok; l++) {
    size_t n = (size_t)BRAIN_IN[l] * BRAIN_OUT[l];
    float* p = (float*)heap_caps_aligned_alloc(16, n * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) { sram_f32_ok = false; break; }
    memcpy(p, BRAIN_F32[l], n * 4);
    W_sram[l] = p;
  }
  if (!sram_f32_ok) {
    for (int l = 0; l < BRAIN_N_LAYERS; l++) if (W_sram[l]) { heap_caps_free((void*)W_sram[l]); W_sram[l] = nullptr; }
  }
  reset_memory();
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  if (!Serial.available()) return;
  int ch = Serial.read();
  if (ch == 0xAA) {
    uint32_t t = millis();
    while (!Serial.available() && millis() - t < 50) {}
    if (Serial.read() != 0x55) return;
    float obs[N_IN], logits[N_ACT];
    if (Serial.readBytes((uint8_t*)obs, sizeof obs) != sizeof obs) return;
    uint8_t a;
    uint32_t us = think(obs, logits, &a);
    uint8_t out[2 + sizeof logits + 1 + 4] = {0x55, 0xAA};
    memcpy(out + 2, logits, sizeof logits);
    out[2 + sizeof logits] = a;
    memcpy(out + 3 + sizeof logits, &us, 4);
    Serial.write(out, sizeof out);
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));  // one blink per decision
  } else if (ch == 'R') {
    reset_memory();
    Serial.println("{\"reset\":true}");
  } else if (ch >= '0' && ch < '0' + N_MODES) {
    Mode m = (Mode)(ch - '0');
    if (m == F32_SRAM_SIMD && !sram_f32_ok) {
      Serial.println("{\"error\":\"float32 weights do not fit in internal RAM\"}");
    } else {
      mode = m;
      info();
    }
  } else if (ch == 'B') {
    bench();
  } else if (ch == 'I') {
    info();
  }
}
