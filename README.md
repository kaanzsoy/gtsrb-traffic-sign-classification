# Traffic Sign Classification under Resource Constraints

This repository includes the code, trained models, result files, and report materials for our traffic sign classification project on the **German Traffic Sign Recognition Benchmark (GTSRB)**.

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
│   └── ResNet50.py
│
├── results/
│   ├── int8_confusion_increase_vs_fp32.csv
│   └── int8_top_confusions_by_worst_classes.csv
│
├── report/
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

## Implemented Experiments

The repository includes implementations for both **MobileNetV2** and **ResNet-50** experiments.

The MobileNetV2 notebook covers:
- FP32 MobileNetV2 training
- ROI-based preprocessing
- INT8 post-training quantization
- FP16 quantization
- Accuracy, ECE, model size, and latency evaluation
- Class-wise and confusion-based analysis

The ResNet-50 implementation covers:
- FP32 ResNet-50 training
- INT8 post-training quantization
- Accuracy evaluation
- Model size measurement
- Inference latency measurement
- Expected Calibration Error (ECE) calculation

The detailed experimental discussion and comparison are provided in the project reports under the `report/` directory.

## How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create and activate a virtual environment

```bash
python -m venv ml_env
source ml_env/bin/activate
```

For Windows:

```bash
ml_env\Scripts\activate
```

### 3. Install required packages

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter opencv-python
```

### 4. Prepare the dataset

The dataset is not included in this repository because of its file size. Download the GTSRB dataset from Kaggle:

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

The `dataset/` folder is excluded from Git tracking, so it must be added manually before running the notebook.

### 5. Run the experiments

For the MobileNetV2 experiments:

```bash
jupyter notebook notebooks/MobileNetV2.ipynb
```

For the ResNet-50 experiments:

```bash
python notebooks/ResNet50.py
```

Then run the notebook/script after placing the dataset under the `dataset/` directory.

The experiments save trained and converted models under:

```text
models/
```

They also save analysis outputs under:

```text
results/
```

## Output Files

The following model files are included or generated during the MobileNetV2 workflow:

```text
mobilenetv2_roi_fp32_best.keras
mobilenetv2_roi_int8.tflite
mobilenetv2_roi_int8_large_rep.tflite
mobilenetv2_roi_fp16.tflite
```

The following result files are generated for INT8 class-wise confusion analysis:

```text
int8_top_confusions_by_worst_classes.csv
int8_confusion_increase_vs_fp32.csv
```

## Notes on Reproducibility

The `dataset/` folder and the local virtual environment folder `ml_env/` are excluded from Git tracking. The dataset should be downloaded manually from Kaggle, and the environment should be recreated using the installation steps above.

Latency measurements may vary depending on hardware, TensorFlow runtime, and numerical format support. Therefore, inference time values should be interpreted with the experimental setup in mind.

## Report

The LaTeX report files are included under the `report/` directory.

The directory contains the first-stage and second-stage project reports, including the LaTeX source files, bibliography files, ICLR style files, and compiled PDF outputs.

## Authors

Hayrettin Kaan Ozsoy<br>
Hacettepe University<br>
hkaanozsoy26@hacettepe.edu.tr

Ahmet Baspinar<br>
Hacettepe University<br>
ahmetbaspinar26@hacettepe.edu.tr

