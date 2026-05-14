# Traffic Sign Classification under Resource Constraints

This repository includes the code, trained models, result files, and report materials for our traffic sign classification project on the **German Traffic Sign Recognition Benchmark (GTSRB) dataset**.

The project investigates how different model choices and quantization formats affect traffic sign classification under resource constraints. Instead of focusing only on **accuracy**, we also evaluate **calibration, model size, inference latency, and class-wise behavior**.

## Project Summary

Traffic sign classification is a widely studied task, but practical deployment requires more than high accuracy. A model used in an embedded or real-time setting should also be small, fast, and reasonably reliable in its confidence estimates.

In this project, the following aspects are evaluated:

- Accuracy
- Expected Calibration Error (ECE)
- Model size
- Inference latency
- Class-wise accuracy degradation
- Confusion behavior after quantization

We worked with **ResNet-50** as a high-capacity baseline and **MobileNetV2** as a lightweight alternative. The main comparison is between FP32 models and lower-precision versions, especially INT8 post-training quantization and FP16 quantization.

## Repository Structure

```text
PROJECT/
│
├── dataset/    # Not included in the repository, download from Kaggle
│   ├── Meta/
│   ├── Test/
│   ├── Train/
│   ├── Meta.csv
│   ├── Test.csv
│   └── Train.csv
│
├── models/
│   ├── mobilenetv2_fp32_best.keras
│   ├── mobilenetv2_roi_fp32_best.keras
│   ├── mobilenetv2_roi_int8.tflite
│   ├── mobilenetv2_roi_int8_large_rep.tflite
│   └── mobilenetv2_roi_fp16.tflite
│
├── notebooks/
│   └── MobileNetV2.ipynb
│
├── results/
│   ├── int8_confusion_increase_vs_fp32.csv
│   └── int8_top_confusions_by_worst_classes.csv
│
└── README.md
```

The `dataset/` folder is shown for clarity, but it is not tracked by Git because of its file size. The dataset should be downloaded manually from Kaggle and placed in this structure before running the notebook.

## Dataset

The experiments use the **German Traffic Sign Recognition Benchmark (GTSRB)**. The dataset contains traffic sign images from 43 classes and includes realistic variations in scale, illumination, viewpoint, and image quality.

The dataset is not included in this repository because of its file size. It can be downloaded from Kaggle:

https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

After downloading, place the dataset inside the `dataset/` folder using the following structure:

```text
dataset/
├── Train/
├── Test/
├── Meta/
├── Train.csv
├── Test.csv
└── Meta.csv
```
The notebook assumes that the dataset folder is located at the project root.


