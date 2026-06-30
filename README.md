# 🍷 Wine Quality Prediction using Machine Learning

## 📌 Project Overview

This project predicts the **quality of red wine** based on its physicochemical properties using supervised machine learning algorithms. It compares the performance of three popular classification models and analyzes which chemical characteristics have the greatest influence on wine quality.

The project includes data exploration, visualization, preprocessing, model training, evaluation, cross-validation, and feature importance analysis.

---

## 🎯 Objectives

- Analyze the physicochemical properties of red wine.
- Predict wine quality scores using machine learning.
- Compare the performance of multiple classification algorithms.
- Identify the most influential chemical features affecting wine quality.
- Visualize relationships between wine properties and quality.

---

## 📂 Dataset

The project uses the **WineQT.csv** dataset.

### Dataset Information

- **Samples:** 1,143 red wine records
- **Features:** 11 physicochemical attributes
- **Target:** Wine quality score (3–8)

### Features

| Feature | Description |
|----------|-------------|
| Fixed Acidity | Tartaric acid concentration |
| Volatile Acidity | Acetic acid concentration |
| Citric Acid | Citric acid content |
| Residual Sugar | Remaining sugar after fermentation |
| Chlorides | Salt concentration |
| Free Sulfur Dioxide | Free SO₂ level |
| Total Sulfur Dioxide | Total SO₂ level |
| Density | Wine density |
| pH | Acidity level |
| Sulphates | Potassium sulphate concentration |
| Alcohol | Alcohol percentage |
| Quality | Wine quality score (Target) |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🤖 Machine Learning Models

The following classification models were trained and compared:

- 🌲 Random Forest Classifier
- 📈 Stochastic Gradient Descent (SGD) Classifier
- 🎯 Support Vector Classifier (SVC)

---

## 📊 Project Workflow

### 1. Data Loading

- Import dataset
- Remove unnecessary identifier column
- Inspect dataset shape
- Check missing values
- Analyze quality score distribution

### 2. Exploratory Data Analysis (EDA)

- Wine quality distribution
- Correlation heatmap
- Chemical property boxplots
- Density vs Fixed Acidity scatter plot
- Feature distribution histograms

### 3. Data Preprocessing

- Feature selection
- Train-Test Split (80:20)
- Standardization using StandardScaler
- Stratified sampling

### 4. Model Training

- Random Forest Classifier
- SGD Classifier
- Support Vector Classifier (RBF Kernel)

### 5. Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score (Macro)
- F1 Score (Weighted)
- Classification Report
- Confusion Matrix
- 5-Fold Cross Validation

### 6. Feature Importance

- Random Forest Feature Importance
- Ranking of the most influential wine characteristics

---

## 📈 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score (Macro)
- F1 Score (Weighted)
- Classification Report
- Confusion Matrix
- 5-Fold Cross Validation

---

## 📊 Visualizations

The project generates several visualizations to better understand the dataset and model performance.

### Exploratory Data Analysis

- 📊 Quality Distribution
- 🔥 Correlation Heatmap
- 📦 Chemical Properties vs Quality
- 📍 Density vs Fixed Acidity Scatter Plot
- 📈 Feature Distribution Histograms

### Model Analysis

- ✅ Confusion Matrices
- 📊 Feature Importance Plot
- 🏆 Model Accuracy Comparison

All figures are automatically saved in the **figures/** directory.

---

## 📁 Project Structure

```
Wine-Quality-Prediction/
│
├── WineQT.csv
├── Wine_Quality_Prediction.py
├── figures/
│   ├── 01_quality_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_chemical_properties_vs_quality.png
│   ├── 04_density_vs_acidity_scatter.png
│   ├── 05_feature_distributions.png
│   ├── 06_confusion_matrices.png
│   ├── 07_feature_importance.png
│   └── 08_model_accuracy_comparison.png
├── model_comparison_summary.csv
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/Wine-Quality-Prediction.git
```

### Navigate to the project directory

```bash
cd Wine-Quality-Prediction
```

### Install the required dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python Wine_Quality_Prediction.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 📌 Key Findings

- Wine quality is an imbalanced multiclass classification problem with scores ranging from **3 to 8**.
- Alcohol content shows one of the strongest positive correlations with wine quality.
- Volatile acidity tends to have a negative relationship with wine quality.
- Random Forest provides feature importance scores, helping identify the most influential chemical properties.
- Model comparison highlights the strengths and weaknesses of each classifier for multiclass prediction.

---

## 🚀 Future Improvements

Potential enhancements include:

- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Ensemble methods such as XGBoost, LightGBM, or CatBoost
- Addressing class imbalance using SMOTE or class weighting
- Feature selection and dimensionality reduction (PCA)
- Model deployment using Flask, FastAPI, or Streamlit
- Building an interactive dashboard for wine quality prediction

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Your Name**

GitHub: https://github.com/yourusername

---

## 🙏 Acknowledgements

- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- UCI Wine Quality Dataset
