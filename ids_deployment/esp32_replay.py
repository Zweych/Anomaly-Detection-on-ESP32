import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

parser = argparse.ArgumentParser(description="ESP32 INT8 PTQ replay evaluator")

parser.add_argument(
    "--mode",
    choices=["pc", "esp32"],
    required=True,
)

parser.add_argument(
    "--dataset",
    choices=["Primary"],
    default="Primary",
)

parser.add_argument(
    "--csv",
    default=None,
)

parser.add_argument(
    "--model",
    default=None,
)

parser.add_argument(
    "--metadata",
    default=None,
)

parser.add_argument(
    "--port",
    default="COM7",
)

parser.add_argument(
    "--baud",
    type=int,
    default=115200,
)

parser.add_argument(
    "--out",
    default=None,
)

parser.add_argument(
    "--metrics-out",
    default=None,
)

parser.add_argument(
    "--limit",
    type=int,
    default=0,
)

parser.add_argument(
    "--timeout",
    type=float,
    default=5.0,
)

args = parser.parse_args()

DATASET_OUT = Path("deployment_artifacts") / args.dataset

if args.csv is None:
    args.csv = str(DATASET_OUT / "esp32_replay.csv")

if args.metadata is None:
    args.metadata = str(DATASET_OUT / "esp32_metadata.json")

if args.model is None:
    args.model = str(DATASET_OUT / "model_esp32_int8_compat.tflite")

if args.out is None:
    args.out = str(
        DATASET_OUT
        / (
            "pc_tflite_int8_compat_predictions.csv"
            if args.mode == "pc"
            else "esp32_int8_compat_predictions.csv"
        )
    )

if args.metrics_out is None:
    args.metrics_out = str(
        DATASET_OUT
        / (
            "pc_tflite_int8_compat_metrics.json"
            if args.mode == "pc"
            else "esp32_int8_compat_metrics.json"
        )
    )

Path(args.out).parent.mkdir(
    parents=True,
    exist_ok=True,
)

Path(args.metrics_out).parent.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURES = [
    "temperature",
    "pressure",
    "humidity",
    "temp_diff",
    "press_diff",
    "hum_diff",
    "temp_roll_std",
    "temp_mean_5",
    "hum_mean_5",
]

CLASS_NAMES = [
    "Normal",
    "Attack",
]

THRESHOLD = 0.50


def load_metadata():
    with open(
        args.metadata,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    if metadata.get("class_mapping", {}).get("0") != "Normal":
        raise RuntimeError("Metadata tidak menunjukkan 0=Normal.")

    if metadata.get("class_mapping", {}).get("1") != "Attack":
        raise RuntimeError("Metadata tidak menunjukkan 1=Attack.")

    if metadata.get("output_activation") != "sigmoid":
        raise RuntimeError("Model metadata bukan model " "sigmoid final.")

    return metadata


def load_replay():
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Replay CSV tidak ditemukan: " f"{args.csv}")

    df = pd.read_csv(args.csv)

    required = [
        "sample_id",
        *FEATURES,
        "true_label",
        "true_class",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError("Kolom replay tidak lengkap: " + ", ".join(missing))

    if args.limit < 0:
        raise ValueError("--limit tidak boleh negatif.")

    if args.limit > 0:
        df = df.iloc[: args.limit].copy()

    if len(df) == 0:
        raise RuntimeError("Replay CSV tidak memiliki sample.")

    df["true_label"] = df["true_label"].astype(int)

    if not df["true_label"].isin([0, 1]).all():
        raise RuntimeError("true_label harus 0=Normal " "atau 1=Attack.")

    return df.reset_index(drop=True)


def calculate_metrics(
    y_true,
    y_pred,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision_attack = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    recall_attack = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1_attack = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_attack": float(precision_attack),
        "recall_attack": float(recall_attack),
        "f1_attack": float(f1_attack),
        "far": float(far),
        "confusion_matrix": {
            "labels": [
                "Normal",
                "Attack",
            ],
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "matrix": cm.tolist(),
        },
    }


def print_metrics(metrics):
    print()
    print("==============================================")
    print("CLASSIFICATION RESULTS")
    print("==============================================")

    print(f"Accuracy          : " f"{metrics['accuracy'] * 100:.4f}%")

    print(f"Precision (Macro) : " f"{metrics['precision_macro'] * 100:.4f}%")

    print(f"Recall (Macro)    : " f"{metrics['recall_macro'] * 100:.4f}%")

    print(f"F1-Score (Macro)  : " f"{metrics['f1_macro'] * 100:.4f}%")

    print(f"Precision Attack  : " f"{metrics['precision_attack'] * 100:.4f}%")

    print(f"Recall Attack     : " f"{metrics['recall_attack'] * 100:.4f}%")

    print(f"F1-Score Attack   : " f"{metrics['f1_attack'] * 100:.4f}%")

    print(f"False Alarm Rate   : " f"{metrics['far'] * 100:.4f}%")

    cm = metrics["confusion_matrix"]

    print()
    print("Confusion Matrix")
    print("                 Predicted")
    print("                 Normal Attack")
    print(f"Actual Normal    " f"{cm['TN']:6d} " f"{cm['FP']:6d}")
    print(f"Actual Attack    " f"{cm['FN']:6d} " f"{cm['TP']:6d}")


def save_metrics(
    metrics,
    sample_count,
    extra=None,
):
    payload = {
        "mode": args.mode,
        "dataset": args.dataset,
        "csv": args.csv,
        "sample_count": int(sample_count),
        "threshold": THRESHOLD,
        "class_mapping": {
            "0": "Normal",
            "1": "Attack",
        },
        "far_definition": "FP / (FP + TN)",
        "model_type": "DNN/MLP INT8 PTQ",
        "output_activation": "sigmoid",
        "output_interpretation": "dequantized probability, no sigmoid",
        "metrics": metrics,
    }

    if extra:
        payload.update(extra)

    with open(
        args.metrics_out,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )


def save_predictions(df_result):
    df_result.to_csv(
        args.out,
        index=False,
    )


def run_pc(
    df_replay,
    metadata,
):
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError(
            "TensorFlow tidak ditemukan. " "Install dengan: " "pip install tensorflow"
        )

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model tidak ditemukan: " f"{args.model}")

    interpreter = tf.lite.Interpreter(model_path=args.model)

    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]

    output_detail = interpreter.get_output_details()[0]

    if input_detail["dtype"] != np.int8:
        raise RuntimeError("Input model bukan INT8.")

    if output_detail["dtype"] != np.int8:
        raise RuntimeError("Output model bukan INT8.")

    input_scale, input_zero_point = input_detail["quantization"]

    output_scale, output_zero_point = output_detail["quantization"]

    mean = np.asarray(
        metadata["standard_scaler_mean"],
        dtype=np.float32,
    )

    scale = np.asarray(
        metadata["standard_scaler_scale"],
        dtype=np.float32,
    )

    X_raw = df_replay[FEATURES].to_numpy(dtype=np.float32)

    X_scaled = (X_raw - mean) / scale

    predictions = []
    probabilities = []
    quantized_outputs = []
    inference_times = []

    for sample in X_scaled:
        q = np.round(sample / input_scale + input_zero_point)

        q = np.clip(
            q,
            -128,
            127,
        ).astype(np.int8)

        interpreter.set_tensor(
            input_detail["index"],
            q.reshape(input_detail["shape"]),
        )

        t0 = time.perf_counter()

        interpreter.invoke()

        elapsed_us = (time.perf_counter() - t0) * 1_000_000.0

        raw_output = interpreter.get_tensor(output_detail["index"]).reshape(-1)[0]

        # Model final menggunakan sigmoid.
        # Jadi output INT8 langsung
        # didequantisasi menjadi probability.
        probability = (float(raw_output) - float(output_zero_point)) * float(
            output_scale
        )

        probability = np.clip(
            probability,
            0.0,
            1.0,
        )

        predicted = 1 if probability >= THRESHOLD else 0

        predictions.append(predicted)

        probabilities.append(float(probability))

        quantized_outputs.append(int(raw_output))

        inference_times.append(elapsed_us)

    result = df_replay.copy()

    result["predicted_label"] = predictions

    result["predicted_class"] = [CLASS_NAMES[x] for x in predictions]

    result["probability_attack"] = probabilities

    result["quantized_output"] = quantized_outputs

    result["inference_us"] = inference_times

    metrics = calculate_metrics(
        result["true_label"].tolist(),
        result["predicted_label"].tolist(),
    )

    metrics["average_tflite_inference_us"] = float(np.mean(inference_times))

    metrics["minimum_tflite_inference_us"] = float(np.min(inference_times))

    metrics["maximum_tflite_inference_us"] = float(np.max(inference_times))

    metrics["input_quantization"] = {
        "scale": float(input_scale),
        "zero_point": int(input_zero_point),
    }

    metrics["output_quantization"] = {
        "scale": float(output_scale),
        "zero_point": int(output_zero_point),
    }

    save_predictions(result)

    save_metrics(
        metrics,
        len(result),
        {
            "model": args.model,
            "quantization": "INT8 PTQ",
        },
    )

    return result, metrics


def run_esp32(df_replay):
    try:
        import serial
    except ImportError:
        raise RuntimeError(
            "pyserial tidak ditemukan. " "Install dengan: " "pip install pyserial"
        )

    try:
        ser = serial.Serial(
            args.port,
            args.baud,
            timeout=0.1,
        )
    except serial.SerialException as exc:
        raise RuntimeError(
            f"Tidak dapat membuka "
            f"{args.port}: {exc}\n"
            "Pastikan Serial Monitor "
            "ditutup."
        )

    print(f"\nSerial connected: " f"{args.port}")

    time.sleep(2.0)
    ser.reset_input_buffer()

    ser.write(b"MODE REPLAY\n")
    ser.flush()

    time.sleep(0.5)

    startup_deadline = time.time() + 2.0

    while time.time() < startup_deadline:
        raw = (
            ser.readline()
            .decode(
                "utf-8",
                errors="ignore",
            )
            .strip()
        )

        if raw:
            print(raw)

    print("\n=== ESP32 REPLAY START ===")

    results = []
    inference_times = []

    try:
        for index, row in df_replay.iterrows():
            sample_id = str(row["sample_id"])

            values = [row[col] for col in FEATURES]

            line = (
                "R,"
                + sample_id
                + ","
                + ",".join(f"{float(v):.10g}" for v in values)
                + ","
                + str(int(row["true_label"]))
                + "\n"
            )

            ser.write(line.encode("utf-8"))

            ser.flush()

            deadline = time.time() + args.timeout

            answer = None

            while time.time() < deadline:
                raw = (
                    ser.readline()
                    .decode(
                        "utf-8",
                        errors="ignore",
                    )
                    .strip()
                )

                if not raw:
                    continue

                if raw.startswith("RESULT,"):
                    parts = raw.split(",")

                    if len(parts) >= 8 and parts[1] == sample_id:
                        answer = parts

                        print(f"[{index + 1}/" f"{len(df_replay)}] " f"{raw}")

                        break

                else:
                    print(raw)

            if answer is None:
                raise RuntimeError(
                    "Timeout menunggu " f"RESULT sample_id=" f"{sample_id}"
                )

            probability = float(answer[2])

            predicted = int(answer[3])

            true_label = int(answer[4])

            inference_us = int(answer[5])

            correct = int(answer[6])

            raw_output = int(answer[8]) if len(answer) >= 9 else None

            inference_times.append(inference_us)

            results.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "true_label": true_label,
                    "true_class": CLASS_NAMES[true_label],
                    "predicted_label": predicted,
                    "predicted_class": CLASS_NAMES[predicted],
                    "probability_attack": probability,
                    "quantized_output": raw_output,
                    "correct": correct,
                    "inference_us": inference_us,
                }
            )

    finally:
        ser.close()

    result = pd.DataFrame(results)

    y_true = result["true_label"].astype(int).tolist()

    y_pred = result["predicted_label"].astype(int).tolist()

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    metrics["average_inference_us"] = float(np.mean(inference_times))

    metrics["minimum_inference_us"] = int(np.min(inference_times))

    metrics["maximum_inference_us"] = int(np.max(inference_times))

    metrics["serial_port"] = args.port

    metrics["baud"] = args.baud

    metrics["output_interpretation"] = (
        "ESP32 output is already sigmoid " "probability after dequantization"
    )

    save_predictions(result)

    save_metrics(
        metrics,
        len(result),
        {
            "hardware": "ESP32",
            "model": "TFLite INT8 PTQ deployed",
        },
    )

    return result, metrics


# ============================================================
# MAIN
# ============================================================

metadata = load_metadata()

df_replay = load_replay()

sample_count = len(df_replay)

THRESHOLD = float(
    metadata.get(
        "threshold",
        THRESHOLD,
    )
)

print("==============================================")

print("ESP32 INT8 PTQ REPLAY EVALUATOR")

print("==============================================")

print(f"Mode       : " f"{args.mode.upper()}")

print(f"Dataset    : " f"{args.dataset}")

print(f"CSV        : " f"{args.csv}")

print(f"Samples    : " f"{sample_count}")

print(f"Threshold  : " f"{THRESHOLD}")

print("Class      : 0=Normal, 1=Attack")

print("Output     : sigmoid probability")

if args.mode == "pc":
    result, metrics = run_pc(
        df_replay,
        metadata,
    )
else:
    result, metrics = run_esp32(df_replay)

print_metrics(metrics)

print()
print("==============================================")

print("OUTPUT FILES")

print("==============================================")

print("Predictions : " f"{args.out}")

print("Metrics     : " f"{args.metrics_out}")

if args.mode == "pc":
    print(
        f"Average TFLite inference : "
        f"{metrics['average_tflite_inference_us']:.2f} us"
    )

if args.mode == "esp32":
    print(f"Average inference : " f"{metrics['average_inference_us']:.2f} us")

    print(f"Minimum inference : " f"{metrics['minimum_inference_us']} us")

    print(f"Maximum inference : " f"{metrics['maximum_inference_us']} us")

print("==============================================")
