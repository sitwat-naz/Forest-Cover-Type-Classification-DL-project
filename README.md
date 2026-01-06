# Forest-Cover-Type-Classification-DL-project

## Project Overview

This project is my first Deep Learning endeavor as part of the **AtomCamp Cohort 15** program, focusing on classifying forest cover types using the **UCI Forest CoverType dataset**. The goal was to build, tune, and evaluate a Neural Network (Multi-Layer Perceptron - MLP) against traditional ensemble methods, aiming for approximately 94% test accuracy.

## Dataset

The **UCI Forest CoverType dataset** contains 581,012 observations with 54 features (e.g., elevation, hydrological features, wilderness areas, and soil types) used to predict 7 distinct forest cover types.

## Objectives Achieved

* **Neural Network Architecture:** Developed a robust 4-layer MLP (`512 -> 256 -> 128 -> 64`) with `ReLU` activations, integrating **Batch Normalization** for training stability.
* **Regularization:** Applied **Dropout (0.3)** and **L2 Regularization** (`1e-4`) to prevent overfitting, crucial for high-dimensional tabular data.
* **Optimization:** Utilized the **Adam optimizer** combined with a `ReduceLROnPlateau` learning rate scheduler for efficient and stable convergence.
* **Training Management:** Implemented **EarlyStopping** (patience=12) to optimize training epochs and prevent overfitting, alongside logging and visualizing training curves.
* **Model Evaluation:** Achieved **94% test accuracy** and reported comprehensive metrics including Precision, Recall, and F1-score (macro and weighted). Performed detailed error analysis using a **Confusion Matrix**.
* **Comparative Analysis:** Trained and evaluated an **XGBoost Classifier** as a benchmark, comparing its performance (83% accuracy) against the optimized MLP.

## Key Insights

* The Deep Learning MLP successfully leveraged the large dataset size (500k+ samples) and careful regularization to achieve a superior performance (94% accuracy) over a baseline XGBoost model (83%).
* Batch Normalization and Dropout were critical in handling the diverse feature scales and preventing the network from overfitting to specific features like Soil Type.
* While overall performance was strong, the Confusion Matrix highlighted common misclassifications between ecologically similar forest types (e.g., Class 0 and Class 1), indicating areas for potential future model refinement.

## Technologies Used

* `Python`
* `TensorFlow / Keras`
* `Scikit-learn`
* `Pandas`
* `Numpy`
* `Matplotlib`
* `Seaborn`
* `XGBoost`

## How to Run

1.  Clone this repository: `git clone [YOUR_REPO_URL]`
2.  Navigate to the project directory.
3.  Install dependencies: `pip install -r requirements.txt` (or list them explicitly if no `requirements.txt`)
4.  Open and run the `[YOUR_NOTEBOOK_NAME].ipynb` notebook.

## Future Work

* Experiment with **class weighting** or **SMOTE** to improve performance on minority classes.
* Explore more advanced ensemble tuning for XGBoost to potentially close the performance gap.
* Investigate feature engineering to create more discriminative features for hard-to-distinguish classes.

---
