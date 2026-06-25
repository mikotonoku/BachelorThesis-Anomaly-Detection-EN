# Anomaly Detection under Limited Anomalous Data

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![Bachelor Thesis](https://img.shields.io/badge/Bachelor-Thesis-success)

## About the Project

This repository contains the source code, data preprocessing pipelines, and experiments developed as part of a bachelor's thesis focused on anomaly detection using machine learning methods under conditions of limited anomalous data.

The objective of this thesis was to analyze and compare the performance of three anomaly detection methods:

* Isolation Forest
* One-Class Support Vector Machine (One-Class SVM)
* Autoencoder

The experiments were conducted on three datasets representing different types of data:

* **NAB (Numenta Anomaly Benchmark)** – industrial time-series data,
* **ECG5000** – biomedical ECG signals,
* **NASA Battery Dataset** – lithium-ion battery degradation data.

In all experiments, the models were trained exclusively on normal data. Anomalous samples were used only during the testing and evaluation stages.

---

## 📑 Table of Contents

* 📘 [Bachelor's Thesis](#-bachelors-thesis)
* 🔗 [Useful Links](#-useful-links)
* 📂 [Repository Structure](#-repository-structure)
* 📊 [Datasets](#-datasets)

  * 🔋 [NASA Battery Dataset](#-nasa-battery-dataset)
  * 🏭 [NAB (Numenta Anomaly Benchmark)](#-nab-numenta-anomaly-benchmark)
  * ❤️ [ECG5000](#️-ecg5000)
* 📓 [Notebook Description](#-notebook-description)
* ⚙️ [Common Experimental Settings](#️-common-experimental-settings)
* 🏆 [Best Results](#-best-results)
* 📈 [Example Experimental Outputs](#-example-experimental-outputs)
* 💻 [Installation](#-installation)
* ▶️ [Running the Project](#️-running-the-project)
* 👩‍🎓 [Author](#author)

---

# 📘 Bachelor's Thesis

A detailed theoretical background, proposed methodology, dataset preprocessing, implementation of the anomaly detection methods, and analysis of the experimental results are presented in the bachelor's thesis.

📄 **Bachelor's Thesis (PDF):**

[BP_RK_IevgeniiaIevgrafova.pdf](docs/BP_RK_IevgeniiaIevgrafova.pdf)

<p align="right">
<a href="#-table-of-contents">⬆️ Back to Table of Contents</a>
</p>

# 🔗 Useful Links

## Development Environment

* [Anaconda](https://www.anaconda.com/download) – Python distribution for data analysis and machine learning.
* [Jupyter Notebook](https://jupyter.org/install) – Interactive environment for running experimental notebooks.
* [Python](https://www.python.org/downloads/) – Programming language used for the implementation.

## Libraries

* [pandas](https://pandas.pydata.org/) – Data manipulation and analysis.
* [NumPy](https://numpy.org/) – Numerical computing.
* [scikit-learn](https://scikit-learn.org/stable/) – Implementation of the Isolation Forest and One-Class SVM algorithms.
* [TensorFlow](https://www.tensorflow.org/install) – Neural network implementation.
* [Keras](https://keras.io/) – Building and training Autoencoder models.
* [Matplotlib](https://matplotlib.org/) – Data visualization.

## Datasets

* [NASA Battery Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)
* [NAB (Numenta Anomaly Benchmark)](https://www.kaggle.com/datasets/boltzmannbrain/nab)
* [ECG5000 Dataset](https://www.kaggle.com/code/mineshjethva/ecg-anomaly-detection)

<p align="right">
<a href="#-table-of-contents">⬆️ Back to Table of Contents</a>
</p>

---

# 📂 Repository Structure

```text
BachelorThesis-Anomaly-Detection-SK
│
├── notebooks/
│   │
│   ├── Autoencoder_ECG5000.ipynb
│   ├── Autoencoder_NAB.ipynb
│   ├── Autoencoder_NASA.ipynb
│   │
│   ├── Isolation_Forest_ECG5000.ipynb
│   ├── Isolation_Forest_NAB.ipynb
│   ├── Isolation_Forest_NASA.ipynb
│   │
│   ├── OneClass_SVM_ECG5000.ipynb
│   ├── OneClass_SVM_NAB.ipynb
│   ├── OneClass_SVM_NASA.ipynb
│   │
│   ├── nab.ipynb
│   └── nasa_battery.ipynb
│
├── data_raw/
│   │
│   ├── ECG5000/
│   ├── nab/
│   └── nasa_battery/
│
├── data_processed/
│   │
│   ├── NAB/
│   └── NASA/
│
├── docs/
│   │
│   └── BP_RK_IevgeniiaIevgrafova.pdf
│
└── README.md
```

<p align="right">
<a href="#-table-of-contents">⬆️ Back to Table of Contents</a>
</p>

---

# 📊 Datasets

## 🔋 NASA Battery Dataset

The NASA Battery Dataset contains measurements collected from lithium-ion batteries throughout their operational lifetime.

During preprocessing, only discharge cycles were used. The original measurements were transformed into derived features describing the battery's condition during the degradation process.

### Input Features

* Voltage_measured
* Current_measured
* Temperature_measured
* Current_load
* Voltage_load
* Time

In addition to the original variables, the following derived features were created:

* first-order differences,
* rolling means,
* rolling standard deviations,
* rolling minimum values,
* rolling maximum values.

Anomalous cycles were identified based on battery capacity. Cycles with a capacity below the 20th percentile were labeled as anomalous.

---

## 🏭 NAB (Numenta Anomaly Benchmark)

The NAB dataset consists of an industrial time series containing temperature measurements collected from industrial equipment.

During preprocessing, the following derived features were generated:

* signal value,
* first-order difference,
* rolling mean,
* rolling standard deviation,
* rolling minimum,
* rolling maximum.

Anomalies were labeled according to the official anomaly windows defined by the NAB benchmark.

---

## ❤️ ECG5000

The ECG5000 dataset contains electrocardiogram (ECG) recordings, where each sample consists of 140 points representing a single heartbeat.

* Class 1 represents normal ECG recordings.
* Classes 2–5 were treated as anomalous.

During the experiments:

* models were trained exclusively on normal samples,
* anomalous samples were used only during testing,
* Z-score normalization based on the training data was applied.

<p align="right">
<a href="#-table-of-contents">⬆️ Back to Table of Contents</a>
</p>

---

# 📓 Notebook Description

## Data Preprocessing

### `nab.ipynb`

This notebook performs:

* loading the NAB dataset,
* labeling anomaly windows,
* generating derived features,
* saving the processed dataset for the experiments.

### `nasa_battery.ipynb`

This notebook performs:

* loading the NASA Battery Dataset,
* selecting discharge cycles,
* generating derived features,
* labeling anomalous cycles,
* saving the processed dataset.

The resulting file:

```text
data_processed/nasa_battery_features_rolling.csv
```

is subsequently used by all experiments conducted on the NASA Battery Dataset.

## Experimental Notebooks

### Isolation Forest

* `Isolation_Forest_ECG5000.ipynb`
* `Isolation_Forest_NAB.ipynb`
* `Isolation_Forest_NASA.ipynb`

The implementation includes model training, anomaly score computation, threshold application, and performance evaluation.

### One-Class SVM

* `OneClass_SVM_ECG5000.ipynb`
* `OneClass_SVM_NAB.ipynb`
* `OneClass_SVM_NASA.ipynb`

The implementation uses the RBF kernel and performs classification based on the decision score.

### Autoencoder

* `Autoencoder_ECG5000.ipynb`
* `Autoencoder_NAB.ipynb`
* `Autoencoder_NASA.ipynb`

The Autoencoder is implemented using TensorFlow/Keras. Anomaly detection is based on the reconstruction error.

<p align="right">
<a href="#-table-of-contents">⬆️ Back to Table of Contents</a>
</p>

---

# ⚙️ Common Experimental Settings

The following principles were applied across all experiments:

* models were trained exclusively on normal data,
* anomalous samples were used only during testing,
* performance was evaluated using Accuracy, Precision, Recall, and F1-score,
* confusion matrices were generated for each experiment,
* score histograms were used to visualize the distribution of anomaly scores.

## Isolation Forest

The following default parameters were used in all Isolation Forest experiments:

* `n_estimators = 300`
* `contamination = "auto"`
* `random_state = 42` for the ECG5000 and NAB datasets
* `seeds = [42, 43, 44, 45, 46]` for the NASA Battery dataset

## One-Class SVM

The following default parameters were used in all One-Class SVM experiments:

* `kernel = "rbf"`
* `nu = 0.05`
* `gamma = "auto"`
* `seeds = [42, 43, 44, 45, 46]` for the NASA Battery dataset

## Autoencoder

The same base architecture was used in all Autoencoder experiments:

* `hidden_dim = 32`
* `encoding_dim = 8`
* `batch_size = 64`
* `learning_rate = 0.001`
* `epochs = 100`
* `patience = 10`
* `validation_split = 0.2`
* `optimizer = Adam`
* `loss function = Mean Squared Error (MSE)`
* `seeds = [42, 43, 44, 45, 46]` for the NASA Battery dataset

<p align="right">
<a href="#-table-of-contents">⬆️ Back to Table of Contents</a>
</p>

---

# 🏆 Best Results

The following table summarizes the best-performing configurations and the corresponding results achieved by each anomaly detection method.

| Method  | Dataset      | Threshold | Train Fraction | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
| ------- | ------------ | --------: | -------------: | -----------: | ------------: | ---------: | -----------: |
| iForest | NASA Battery |     0.065 |           0.90 |        85.28 |         85.58 |      94.09 |        89.54 |
| iForest | ECG5000      |       0.0 |              – |        93.42 |         90.09 |      94.61 |        92.29 |
| iForest | NAB          |     -0.15 |           0.80 |        99.29 |         96.68 |      97.71 |        97.19 |
| OCSVM   | NASA Battery |      20.0 |           0.90 |        78.86 |         87.85 |      79.42 |        83.39 |
| OCSVM   | ECG5000      |      -0.3 |              – |        96.87 |         94.59 |      98.08 |        96.30 |
| OCSVM   | NAB          |      -5.0 |           0.80 |        97.59 |         90.89 |      89.77 |        90.33 |
| AE      | NASA Battery |     0.002 |           0.90 |        86.91 |         86.63 |      95.75 |        90.84 |
| AE      | ECG5000      |       0.7 |              – |        94.53 |         94.43 |      92.31 |        93.36 |
| AE      | NAB          |   0.00006 |           0.80 |        95.85 |         80.22 |      88.71 |        84.25 |

<p align="right">
<a href="#-table-of-contents">⬆️ Back to Table of Contents</a>
</p>

---

# 📈 Example Experimental Outputs

The following example presents the results obtained using the **Autoencoder** on the **ECG5000** dataset with the following decision threshold:

```text
Threshold = 0.7
```

### Metrics Table

<img width="1455" height="720" alt="metrics_table" src="https://github.com/user-attachments/assets/f2c3491c-e7c8-4def-b5dc-d55e8c933367" />

### Confusion Matrix

<img width="1753" height="1385" alt="confusion_matrix" src="https://github.com/user-attachments/assets/da42f92c-027d-43b4-82ce-505a46988ca0" />

### Reconstruction Error Histogram

<img width="2967" height="1470" alt="reconstruction_error_histogram" src="https://github.com/user-attachments/assets/c4a6ada0-98c9-42d2-ae01-b7f14df2e169" />

### Reconstruction of Normal and Anomalous ECG Signals

<img width="2969" height="1769" alt="ecg_reconstruction_examples" src="https://github.com/user-attachments/assets/653ece48-bb41-4944-87fc-268187dd8bc2" />

The reconstruction error histogram demonstrates the separation of normal and anomalous samples using the selected decision threshold.

The ECG reconstruction plots illustrate the Autoencoder's ability to reconstruct normal heartbeats more accurately, while anomalous samples exhibit significantly higher reconstruction errors.

<p align="right">
<a href="#-table-of-contents">⬆️ Back to Table of Contents</a>
</p>

---

# 💻 Installation

The project was developed using:

* Python 3.x
* Jupyter Notebook
* Anaconda

Required libraries:

* pandas
* numpy
* scikit-learn
* tensorflow
* keras
* matplotlib
* seaborn
* scipy
* pathlib

<p align="right">
<a href="#-table-of-contents">⬆️ Back to Table of Contents</a>
</p>

---

# ▶️ Running the Project

## NASA Battery

The NASA Battery Dataset is **not included** in this repository due to its size.

Before running the experiments:

1. Download the NASA Battery Dataset.
2. Copy its contents into:

```text
data_raw/nasa_battery/
```

3. Run:

```text
notebooks/nasa_battery.ipynb
```

The notebook will automatically generate:

```text
data_processed/nasa_battery_features_rolling.csv
```

After that, you can run the experimental notebooks for the NASA Battery dataset.

### Note

The `data_processed` directory must exist.

If the `nasa_battery_features_rolling.csv` file is missing, it will be generated automatically by the notebook.

## NAB

The NAB dataset is already included in the repository.

To generate the processed dataset, run:

```text
notebooks/nab.ipynb
```

After preprocessing, the experimental notebooks can be executed.

## ECG5000

The ECG5000 dataset is already included in the repository.

No separate preprocessing step is required.

The experimental notebooks load the dataset directly.

<p align="right">
<a href="#-table-of-contents">⬆️ Back to Table of Contents</a>
</p>

---

<a id="author"></a>

# 👩‍🎓 Author

**Ievgeniia Ievgrafova**

Bachelor's Thesis

Faculty of Electrical Engineering and Information Technology

Slovak University of Technology in Bratislava

2026
