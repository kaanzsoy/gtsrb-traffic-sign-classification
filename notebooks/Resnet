import os
import time
import random
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, applications

#Dataset
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import os
import time


# 1. DATASET PATHS
DATASET_DIR = Path("dataset")

TRAIN_CSV_PATH = DATASET_DIR / "Train.csv"
TEST_CSV_PATH = DATASET_DIR / "Test.csv"

TRAIN_IMAGES_DIR = DATASET_DIR / "Train"
TEST_IMAGES_DIR = DATASET_DIR / "Test"


trainDf = pd.read_csv(TRAIN_CSV_PATH)
testDf = pd.read_csv(TEST_CSV_PATH)

IMG_SIZE = (32, 32)


def load_images(df, img_dir):
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = img_dir / row["Path"]     # CSV kolon adı önemli
        label = row["ClassId"]

        img = cv2.imread(str(img_path))
        img = cv2.resize(img, IMG_SIZE)

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

train_images, train_labels = load_images(trainDf, TRAIN_IMAGES_DIR)
test_images, test_labels = load_images(testDf, TEST_IMAGES_DIR)


# 2. NORMALIZATION
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0


# 3. MODEL (ResNet50)

base_model = applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(43, activation='softmax')  # GTSRB = 43 class
])


# 4. COMPILE
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# 5. TRAIN
model.fit(
    train_images[:5000],
    train_labels[:5000],
    epochs=10,
    validation_split=0.2
)


# 6. EVALUATION
print("\nTest Result:")
test_loss, test_acc = model.evaluate(
    test_images[:1000],
    test_labels[:1000],
    verbose=2
)


# 7. MODEL SIZE
model.save("temp_model.h5")
size_mb = os.path.getsize("temp_model.h5") / (1024 * 1024)
print(f"Model Boyutu: {size_mb:.2f} MB")


# 8. INFERENCE TIME
start_time = time.time()

for i in range(100):
    _ = model.predict(test_images[i:i+1], verbose=0)

end_time = time.time()

avg_inference = (end_time - start_time) / 100 * 1000
print(f"Ortalama Tahmin Süresi: {avg_inference:.2f} ms / görsel")

# 8. EXPECTED CALIBRATION ERROR
def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true.flatten())

    ece = 0
    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.any(bin_mask):
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            bin_weight = np.mean(bin_mask)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece

y_probs = model.predict(test_images[:1000])
ece_score = calculate_ece(test_labels[:1000], y_probs)

print(f"ECE: {ece_score:.4f}")

#############################
# 8 bit QUANTİZATİON SECTİON
###########################

# =========================
# 1. MODEL
# =========================
base_model = applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(43, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images[:10000], train_labels[:10000], epochs=5, validation_split=0.1)

# =========================
# 2. REPRESENTATIVE DATA
# =========================
def representative_data_gen():
    for i in range(100):
        yield [train_images[i:i+1]]

# =========================
# 3. TFLITE INT8 CONVERSION
# =========================
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("model_int8.tflite", "wb") as f:
    f.write(tflite_model)

# =========================
# 4. EVALUATİON
# =========================
def evaluate_tflite(model_path, test_images, test_labels):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    inference_times = []

    n = 500

    for i in range(n):
        input_data = (test_images[i:i+1] * 255).astype(np.int8)

        start = time.perf_counter()

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        inference_times.append(time.perf_counter() - start)

        predictions.append(np.argmax(output))

    acc = np.mean(np.array(predictions) == test_labels[:n].flatten())
    avg_time = np.mean(inference_times) * 1000
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    return acc, avg_time, size_mb
