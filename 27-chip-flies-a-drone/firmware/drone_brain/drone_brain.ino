// Chip is the drone's brain: a flying policy (MLP) answering the MuJoCo sim on the Mac over USB.
//
// Protocol (binary, little endian):
//   Mac -> Chip   AA 55 + 27 x float32 observations
//   Chip -> Mac   55 AA + 4 x float32 motor commands + uint32 inference microseconds
// Text commands (one byte):
//   '0'..'4'  pick how the brain runs (see MODES), answered with a JSON line
//   'B'       benchmark every mode (200 runs each), JSON lines
//   'I'       info: policy size, free RAM, current mode
//
// The policy header comes from sim/export_policy.py. Pick it with -DPOLICY_HEADER or edit below.

#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include "esp_dsp.h"

#ifndef POLICY_HEADER
#define POLICY_HEADER "policy_main.h"
#endif
#include POLICY_HEADER

#ifdef POLICY_N_OBS
#define N_OBS POLICY_N_OBS
#else
#define N_OBS 27
#endif
#define N_ACT 4
#define MAX_W 256

enum Mode { F32_FLASH = 0, F32_PSRAM, F32_PSRAM_SIMD, F32_SRAM_SIMD, I8_SRAM, N_MODES };
static const char* MODE_NAME[N_MODES] = {
    "float32, weights read from flash, plain C",
    "float32, weights in PSRAM, plain C",
    "float32, weights in PSRAM, SIMD dot product",
    "float32, weights in internal RAM, SIMD dot product",
    "int8 weights in internal RAM, plain C",
};

static const float* W_psram[POLICY_N_LAYERS];
static const float* W_sram[POLICY_N_LAYERS];  // only if it fits
static const int8_t* W_i8[POLICY_N_LAYERS];
static bool sram_f32_ok = true;
static Mode mode = F32_PSRAM_SIMD;

static float bufA[MAX_W] __attribute__((aligned(16)));
static float bufB[MAX_W] __attribute__((aligned(16)));
static int8_t qbuf[MAX_W] __attribute__((aligned(16)));

static inline float swish(float x) { return x / (1.0f + expf(-x)); }

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

// int8 weights (per output channel scale), activations quantised on the fly (one scale per layer)
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

static uint32_t run_policy(const float* obs, float* act) {
  int64_t t0 = esp_timer_get_time();
  float* x = bufA;
  float* y = bufB;
  for (int i = 0; i < N_OBS; i++) x[i] = (obs[i] - OBS_MEAN[i]) * OBS_RECIP[i];
  for (int l = 0; l < POLICY_N_LAYERS; l++) {
    int nin = POLICY_DIMS[l], nout = POLICY_DIMS[l + 1];
    switch (mode) {
      case F32_FLASH:      dense_f32(POLICY_F32[l], POLICY_B[l], x, y, nin, nout, false); break;
      case F32_PSRAM:      dense_f32(W_psram[l], POLICY_B[l], x, y, nin, nout, false); break;
      case F32_PSRAM_SIMD: dense_f32(W_psram[l], POLICY_B[l], x, y, nin, nout, true); break;
      case F32_SRAM_SIMD:  dense_f32(W_sram[l], POLICY_B[l], x, y, nin, nout, true); break;
      default:             dense_i8(W_i8[l], POLICY_S[l], POLICY_B[l], x, y, nin, nout); break;
    }
    if (l < POLICY_N_LAYERS - 1) {
      for (int o = 0; o < nout; o++) y[o] = POLICY_SWISH ? swish(y[o]) : fmaxf(y[o], 0.0f);
    }
    float* t = x; x = y; y = t;
  }
  for (int i = 0; i < N_ACT; i++) act[i] = tanhf(x[i]);
  return (uint32_t)(esp_timer_get_time() - t0);
}

static void info() {
  Serial.printf("{\"params\":%d,\"float32_kib\":%.0f,\"int8_kib\":%.0f,\"mode\":%d,\"mode_name\":\"%s\","
                "\"sram_f32_fits\":%s,\"free_internal\":%u,\"free_psram\":%u}\n",
                POLICY_PARAMS, POLICY_PARAMS * 4 / 1024.0, POLICY_PARAMS / 1024.0, mode, MODE_NAME[mode],
                sram_f32_ok ? "true" : "false",
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

static void bench() {
  float obs[N_OBS], act[N_ACT];
  for (int i = 0; i < N_OBS; i++) obs[i] = 0.1f * sinf(i);
  Mode keep = mode;
  for (int m = 0; m < N_MODES; m++) {
    if (m == F32_SRAM_SIMD && !sram_f32_ok) {
      Serial.printf("{\"mode\":%d,\"name\":\"%s\",\"skipped\":\"does not fit in internal RAM\"}\n", m, MODE_NAME[m]);
      continue;
    }
    mode = (Mode)m;
    run_policy(obs, act);  // warm the cache
    uint32_t best = UINT32_MAX, sum = 0;
    for (int k = 0; k < 200; k++) {
      uint32_t us = run_policy(obs, act);
      sum += us;
      if (us < best) best = us;
    }
    Serial.printf("{\"mode\":%d,\"name\":\"%s\",\"mean_us\":%.1f,\"best_us\":%u,\"budget_pct\":%.1f}\n",
                  m, MODE_NAME[m], sum / 200.0, best, sum / 200.0 / 100.0);  // % of a 10 ms (100 Hz) step
  }
  mode = keep;
}

void setup() {
  Serial.setRxBufferSize(1024);
  Serial.begin(921600);
  for (int l = 0; l < POLICY_N_LAYERS; l++) {
    size_t n = (size_t)POLICY_DIMS[l] * POLICY_DIMS[l + 1];
    float* p = (float*)heap_caps_aligned_alloc(16, n * 4, MALLOC_CAP_SPIRAM);
    memcpy(p, POLICY_F32[l], n * 4);
    W_psram[l] = p;
    int8_t* q = (int8_t*)heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    memcpy(q, POLICY_I8[l], n);
    W_i8[l] = q;
  }
  for (int l = 0; l < POLICY_N_LAYERS && sram_f32_ok; l++) {
    size_t n = (size_t)POLICY_DIMS[l] * POLICY_DIMS[l + 1];
    float* p = (float*)heap_caps_aligned_alloc(16, n * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) { sram_f32_ok = false; break; }
    memcpy(p, POLICY_F32[l], n * 4);
    W_sram[l] = p;
  }
  if (!sram_f32_ok) {
    for (int l = 0; l < POLICY_N_LAYERS; l++) if (W_sram[l]) { heap_caps_free((void*)W_sram[l]); W_sram[l] = nullptr; }
  }
  pinMode(LED_BUILTIN, OUTPUT);
}

static uint32_t steps = 0;

void loop() {
  if (!Serial.available()) return;
  int c = Serial.read();
  if (c == 0xAA) {
    // wait for 0x55 then 27 floats
    uint32_t t = millis();
    while (!Serial.available() && millis() - t < 50) {}
    if (Serial.read() != 0x55) return;
    float obs[N_OBS], act[N_ACT];
    if (Serial.readBytes((uint8_t*)obs, sizeof obs) != sizeof obs) return;
    uint32_t us = run_policy(obs, act);
    uint8_t out[2 + sizeof act + 4] = {0x55, 0xAA};
    memcpy(out + 2, act, sizeof act);
    memcpy(out + 2 + sizeof act, &us, 4);
    Serial.write(out, sizeof out);
    if (++steps % 50 == 0) digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));  // blinks once a second at 100 Hz
  } else if (c >= '0' && c < '0' + N_MODES) {
    Mode m = (Mode)(c - '0');
    if (m == F32_SRAM_SIMD && !sram_f32_ok) {
      Serial.println("{\"error\":\"float32 weights do not fit in internal RAM\"}");
    } else {
      mode = m;
      info();
    }
  } else if (c == 'B') {
    bench();
  } else if (c == 'I') {
    info();
  }
}
