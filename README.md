# ML-Based Relative Density Predictor for LPBF

## Overview

This project uses Machine Learning to predict the **Relative Density (%)** of parts manufactured using **Laser Powder Bed Fusion (LPBF)**. The prediction is based on two important process parameters while keeping other process parameters (Hatching Spacing (µm) and Layer Thickness (µm)) constant:

- Laser Power (W)
- Scan Speed (m/s)

The model helps estimate the relative density of LPBF-produced parts without requiring extensive experimental trials, saving both time and resources.

---

## Problem Statement

In Laser Powder Bed Fusion, the quality and density of manufactured parts are strongly influenced by process parameters. Determining the optimal parameter combination experimentally can be expensive and time-consuming.

This project aims to develop a machine learning model capable of predicting relative density from process parameters (Laser Power (W) and Scan Speed (m/s)), helping researchers and engineers make informed decisions before conducting physical experiments.

---

## Features

- Predicts Relative Density (%) using Machine Learning
- Uses Gaussian Process Regression (GPR)
- Provides Prediction Uncertainty
- Accepts User Input for New Parameters
- Generates Contour Plots for Visualization
- Generates 3D Surface Plots
- Supports LPBF Materials: AlSi10Mg, Ti6Al4V and IN718

---

## Dataset

The dataset consists of experimental LPBF data collected from published research literature.

### Input Features

| Feature | Unit |
|----------|----------|
| Laser Power | W |
| Scan Speed | m/s |

### Target Variable

| Output | Unit |
|----------|----------|
| Relative Density | % |

---

## Machine Learning Model

The project uses a Gaussian Process Regression (GPR) model.

### Workflow

1. Load experimental LPBF data
2. Prepare input and output variables
3. Train the Gaussian Process Regression model
4. Predict relative density for new process parameters
5. Estimate prediction uncertainty
6. Visualize results using contour and 3D plots

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AnkeshParashar/ML-Based-Relative-Density-Predictor.git
```

Move to the project folder:

```bash
cd ML-Based-Relative-Density-Predictor
```

Install required libraries:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

---

## Required Libraries

- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-Learn

---

## Usage

Run the Python script:

```bash
python main.py
```

Example Input:

```text
New Laser Power(W): 250
New Scan Speed(m/s): 1.5
```

Example Output:

```text
Predicted Density(%): 98.42
Uncertainty(%): ±0.56
```

---

## Visualization

The project generates several visualizations:

### Experimental Data Plot

Displays the original LPBF dataset.

### Contour Plot

Shows predicted density variation across different laser power and scan speed combinations.

### 3D Surface Plot

Visualizes the relationship between process parameters and relative density.

### Prediction Point

Displays the user-input parameter combination on the generated plots.

---

## Applications

- Additive Manufacturing Research
- LPBF Process Optimization
- Material Development
- Process Parameter Selection
- Academic Projects

---

## Future Work

- Include additional process parameters
- Support more LPBF materials
- Develop a graphical user interface (GUI)
- Deploy as a web application
- Expand the dataset using additional literature sources

---

## Results

The Gaussian Process Regression model successfully predicts relative density and provides uncertainty estimates, making it a useful tool for LPBF process parameter exploration and optimization.

---

## Author

**Ankesh Parashar**

B.Tech Mechanical Engineering, NIT Patna

This project was developed to explore the application of Machine Learning techniques in Additive Manufacturing and Laser Powder Bed Fusion process optimization.

---

## License

This project is intended for educational and research purposes.