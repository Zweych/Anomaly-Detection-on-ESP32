#include <Arduino.h>
#include <math.h>
#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

#include "model_esp32_int8_compat.h"
#include "esp32_scaler.h"
#include "esp32_model_config.h"

#define NUMBER_OF_INPUTS 9
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE (16 * 1024)

Eloquent::TinyML::TensorFlow::TensorFlow<
  NUMBER_OF_INPUTS,
  NUMBER_OF_OUTPUTS,
  TENSOR_ARENA_SIZE>
  ml;

constexpr float EVAL_THRESHOLD = 0.50f;

uint64_t inference_count = 0;
uint64_t total_inference_us = 0;

uint64_t replay_count = 0;
uint64_t replay_correct = 0;

uint32_t ready_free_heap = 0;
uint32_t min_replay_free_heap = UINT32_MAX;
uint32_t max_replay_used_heap = 0;

void print_memory_profile(
  const char *stage) {
  Serial.println();
  Serial.print("=== MEMORY: ");
  Serial.print(stage);
  Serial.println(" ===");

  Serial.print("Free heap     : ");
  Serial.print(
    ESP.getFreeHeap());
  Serial.println(" bytes");

  Serial.print("Min free heap : ");
  Serial.print(
    ESP.getMinFreeHeap());
  Serial.println(" bytes");

  Serial.print("Max alloc     : ");
  Serial.print(
    ESP.getMaxAllocHeap());
  Serial.println(" bytes");

  Serial.print("Tensor arena  : ");
  Serial.print(
    TENSOR_ARENA_SIZE / 1024);
  Serial.println(" KB");

  Serial.print("Model size    : ");
  Serial.print(
    g_model_len);
  Serial.println(" bytes");

  Serial.println(
    "====================");
}

int8_t quantize_input(
  float value) {
  float q =
    roundf(
      value / ESP32_INPUT_SCALE
      + (float)
        ESP32_INPUT_ZERO_POINT);

  if (q > 127.0f)
    q = 127.0f;

  if (q < -128.0f)
    q = -128.0f;

  return (
    int8_t)q;
}

float dequantize_output(
  int8_t value) {
  return (
    (
      (float)value
      - (float)
        ESP32_OUTPUT_ZERO_POINT)
    * ESP32_OUTPUT_SCALE);
}

float sigmoid(
  float x) {
  if (x >= 0.0f) {
    float z =
      expf(-x);

    return (
      1.0f / (1.0f + z));
  }

  float z =
    expf(x);

  return (
    z / (1.0f + z));
}

void scale_features(
  const float *raw,
  float *scaled) {
  for (
    int i = 0;
    i < NUMBER_OF_INPUTS;
    i++) {
    float scale =
      ESP32_SCALER_SCALE[i];

    if (
      fabsf(scale)
      < 1e-12f) {
      scaled[i] = 0.0f;
    } else {
      scaled[i] =
        (raw[i]
         - ESP32_SCALER_MEAN[i])
        / scale;
    }
  }
}

float run_inference(
  const float raw_features[NUMBER_OF_INPUTS],
  uint32_t *inference_us,
  int8_t *raw_output) {
  float scaled[NUMBER_OF_INPUTS];

  scale_features(
    raw_features,
    scaled);

  int8_t quantized[NUMBER_OF_INPUTS];

  for (
    int i = 0;
    i < NUMBER_OF_INPUTS;
    i++) {
    quantized[i] =
      quantize_input(
        scaled[i]);
  }

  int8_t output[NUMBER_OF_OUTPUTS] = { 0 };

  uint32_t start =
    micros();

  /*
   * Direct INT8 inference.
   *
   * Jangan menggunakan float input.
   * Model yang dipakai adalah full INT8.
   */
  ml.predict(
    quantized,
    output);

  *inference_us =
    micros() - start;

  if (!ml.isOk()) {
    Serial.print(
      "ERROR,TINYML,");

    Serial.println(
      ml.getErrorMessage());

    return -1.0f;
  }

  /*
   * Ambil output INT8 langsung
   * dari output buffer.
   */
  *raw_output =
    output[0];

  inference_count++;
  total_inference_us +=
    *inference_us;

  float logit =
    dequantize_output(
      *raw_output);

  float probability =
    sigmoid(logit);

  if (
    isnan(probability)
    || isinf(probability)) {
    Serial.println(
      "ERROR,INVALID_PROBABILITY");

    return -1.0f;
  }

  if (probability < 0.0f)
    probability = 0.0f;

  if (probability > 1.0f)
    probability = 1.0f;

  return probability;
}

bool process_replay_line(
  String line) {
  /*
   * Format:
   *
   * R,sample_id,
   * 9 features,
   * true_label
   *
   * Total token = 12
   */

  if (
    !line.startsWith("R,")) {
    return false;
  }

  String token[12];

  int token_count = 0;
  int start = 0;

  while (
    token_count < 12) {
    int comma =
      line.indexOf(
        ',',
        start);

    if (comma < 0) {
      token[token_count++] =
        line.substring(
          start);

      break;
    }

    token[token_count++] =
      line.substring(
        start,
        comma);

    start =
      comma + 1;
  }

  if (
    token_count != 12) {
    Serial.print(
      "ERROR,BAD_REPLAY_FORMAT,");

    Serial.println(
      token_count);

    return false;
  }

  String sample_id =
    token[1];

  float features[NUMBER_OF_INPUTS];

  for (
    int i = 0;
    i < NUMBER_OF_INPUTS;
    i++) {
    features[i] =
      token[2 + i].toFloat();
  }

  int true_label =
    token[11].toInt();

  if (
    true_label != 0
    && true_label != 1) {
    Serial.print(
      "ERROR,BAD_TRUE_LABEL,");

    Serial.println(
      true_label);

    return false;
  }

  uint32_t inference_us =
    0;

  int8_t raw_output =
    0;

  float probability =
    run_inference(
      features,
      &inference_us,
      &raw_output);

  if (
    probability < 0.0f) {
    return false;
  }

  int predicted =
    probability >= EVAL_THRESHOLD
      ? 1
      : 0;

  bool correct =
    predicted == true_label;

  replay_count++;

  if (correct) {
    replay_correct++;
  }

  uint32_t free_heap =
    ESP.getFreeHeap();

  if (
    free_heap < min_replay_free_heap) {
    min_replay_free_heap =
      free_heap;
  }

  uint32_t used_heap =
    ready_free_heap > free_heap
      ? ready_free_heap - free_heap
      : 0;

  if (
    used_heap > max_replay_used_heap) {
    max_replay_used_heap =
      used_heap;
  }

  float running_accuracy =
    replay_count > 0
      ? 100.0f * (float)replay_correct
          / (float)
               replay_count
      : 0.0f;

  /*
   * RESULT format:
   *
   * RESULT,
   * sample_id,
   * probability,
   * predicted,
   * true_label,
   * inference_us,
   * correct,
   * running_accuracy,
   * raw_output
   */

  Serial.print(
    "RESULT,");

  Serial.print(
    sample_id);

  Serial.print(",");

  Serial.print(
    probability,
    8);

  Serial.print(",");

  Serial.print(
    predicted);

  Serial.print(",");

  Serial.print(
    true_label);

  Serial.print(",");

  Serial.print(
    inference_us);

  Serial.print(",");

  Serial.print(
    correct ? 1 : 0);

  Serial.print(",");

  Serial.print(
    running_accuracy,
    4);

  Serial.print(",");

  Serial.println(
    raw_output);

  return true;
}

void reset_profile() {
  replay_count = 0;
  replay_correct = 0;

  min_replay_free_heap =
    UINT32_MAX;

  max_replay_used_heap =
    0;

  Serial.println(
    "STATE_RESET");
}

void setup_tinyml() {
  Serial.println(
    "Initializing TinyML...");

  print_memory_profile(
    "BEFORE MODEL");

  ml.begin(
    g_model);

  if (!ml.isOk()) {
    Serial.print(
      "ERROR,TINYML_INIT,");

    Serial.println(
      ml.getErrorMessage());

    while (true) {
      delay(1000);
    }
  }

  Serial.println(
    "TinyML DNN INT8 COMPAT model OK");

  Serial.print(
    "Input scale      : ");

  Serial.println(
    ESP32_INPUT_SCALE,
    12);

  Serial.print(
    "Input zero point : ");

  Serial.println(
    ESP32_INPUT_ZERO_POINT);

  Serial.print(
    "Output scale     : ");

  Serial.println(
    ESP32_OUTPUT_SCALE,
    12);

  Serial.print(
    "Output zero point: ");

  Serial.println(
    ESP32_OUTPUT_ZERO_POINT);

  print_memory_profile(
    "AFTER MODEL");
}

void setup() {
  Serial.begin(
    115200);

  delay(1500);

  Serial.println();

  Serial.println(
    "==============================");

  Serial.println(
    "ESP32 DNN/MLP INT8 PAPER TEST");

  Serial.println(
    "TFLite INT8 COMPAT DEPLOYMENT");

  Serial.println(
    "==============================");

  print_memory_profile(
    "BOOT");

  setup_tinyml();

  ready_free_heap =
    ESP.getFreeHeap();

  print_memory_profile(
    "READY");

  Serial.print(
    "READY FREE HEAP: ");

  Serial.println(
    ready_free_heap);

  Serial.println();

  Serial.println(
    "Commands:");

  Serial.println(
    "MODE REPLAY");

  Serial.println(
    "PROFILE");

  Serial.println(
    "RESET");
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String line =
    Serial.readStringUntil(
      '\n');

  line.trim();

  if (
    line.equalsIgnoreCase(
      "MODE REPLAY")) {
    reset_profile();

    Serial.println(
      "MODE=REPLAY");

    return;
  }

  if (
    line.equalsIgnoreCase(
      "PROFILE")) {
    print_memory_profile(
      "MANUAL");

    Serial.print(
      "Ready free heap: ");

    Serial.println(
      ready_free_heap);

    Serial.print(
      "Replay min free heap: ");

    if (
      min_replay_free_heap
      == UINT32_MAX) {
      Serial.println(
        "N/A");
    } else {
      Serial.println(
        min_replay_free_heap);
    }

    Serial.print(
      "Replay max used heap: ");

    Serial.println(
      max_replay_used_heap);

    Serial.print(
      "Inference count: ");

    Serial.println(
      inference_count);

    if (
      inference_count > 0) {
      Serial.print(
        "Average inference us: ");

      Serial.println(
        (
          double)
            total_inference_us
          / (double)
            inference_count,
        2);
    }

    if (
      replay_count > 0) {
      Serial.print(
        "Replay count: ");

      Serial.println(
        replay_count);

      Serial.print(
        "Replay correct: ");

      Serial.println(
        replay_correct);

      Serial.print(
        "Replay accuracy: ");

      Serial.print(
        100.0 * (double)replay_correct
          / (double)
            replay_count,
        4);

      Serial.println("%");
    }

    return;
  }

  if (
    line.equalsIgnoreCase(
      "RESET")) {
    reset_profile();
    return;
  }

  if (
    line.startsWith("R,")) {
    process_replay_line(
      line);

    return;
  }
}