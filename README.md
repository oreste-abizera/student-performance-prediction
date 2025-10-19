# Student Performance Prediction: ML & DL Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Mission](#project-mission)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [License](#license)

---

## 🎯 Overview

This project demonstrates a comprehensive machine learning and deep learning pipeline for predicting student academic performance and identifying high-performing vs. at-risk students. The analysis focuses on bridging education gaps for rural students by understanding and quantifying key success factors.

**11 systematic experiments** are conducted, comparing:
- **6 Traditional ML Models**: Linear Regression, Random Forest, Gradient Boosting, Logistic Regression, Random Forest Classifier, XGBoost
- **5 Deep Learning Models**: Sequential (Simple/Advanced), Functional API (Multi-Path), Classification (Simple/Advanced)

---

## 🌍 Project Mission

> **To bridge the education gap for rural students by providing access to high-quality learning materials and technologies, and fostering their potential through exchange programs in advanced regions to inspire them to achieve great things and uplift their communities.**

This research operationalizes that mission by:
1. **Quantifying the education gap** (LearningAccessIndex as critical predictor)
2. **Identifying at-risk students early** (92%+ accuracy in classification)
3. **Discovering modifiable success factors** (Engagement, Access, Behavior > Demographics)
4. **Enabling targeted interventions** (3-tier framework for resource allocation)
5. **Supporting exchange program selection** (Identifying high performers for advancement)

---

## 📊 Dataset

**Source:** [Zenodo - Student Performance and Learning Behavior Dataset](https://zenodo.org/records/16459132)

| Metric                  | Value           |
| ----------------------- | --------------- |
| **Student Records**     | 14,003          |
| **Original Features**   | 16              |
| **Engineered Features** | 4               |
| **Total Features**      | 20              |
| **Missing Values**      | 0               |
| **Time Period**         | Single snapshot |

### Features by Category

#### Study Behaviors & Engagement (6)
- `StudyHours` - Weekly study time
- `Attendance` - Class attendance percentage
- `Extracurricular` - Extracurricular participation
- `AssignmentCompletion` - Assignment completion rate
- `OnlineCourses` - Online course engagement
- `Discussions` - Discussion participation

#### Resource Access & Learning Environment (3)
- `Resources` - Availability of study materials
- `Internet` - Internet connectivity quality
- `EduTech` - Educational technology access

#### Motivation & Psychological Factors (2)
- `Motivation` - Student motivation level
- `StressLevel` - Perceived academic stress

#### Demographics (2)
- `Gender` - Student gender
- `Age` - Student age (18-30 years)

#### Performance Indicators (2)
- `ExamScore` - Standardized exam score (0-100)
- `FinalGrade` - Overall course grade (0-100)

### Engineered Features

1. **EngagementIndex** = (Attendance + Extracurricular + Discussions + OnlineCourses) / 4
   - Captures student participation and involvement

2. **LearningAccessIndex** = (Resources + Internet + EduTech) / 3
   - Quantifies rural education access gap

3. **BehaviorScore** = [(StudyHours + AssignmentCompletion + Motivation) / 3] - (0.1 × StressLevel)
   - Measures academic discipline and mental health

4. **PerformancePotential** = BehaviorScore × LearningAccessIndex
   - Captures interaction: will × opportunity

---

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Step 1: Clone Repository
```bash
git clone https://github.com/oreste-abizera/student-performance-prediction.git
cd student-performance-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n student-perf python=3.8
conda activate student-perf
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow, sklearn, pandas; print('✓ All packages installed successfully')"
```

---

## 🚀 Quick Start

### Run the Complete Analysis

```bash
# Start Jupyter notebook
jupyter notebook Educational_Performance_ML_DL_Pipeline.ipynb

# Or use Google Colab (no installation needed)
# Open: https://colab.research.google.com/github/oreste-abizera/student-performance-prediction/blob/main/Educational_Performance_ML_DL_Pipeline.ipynb
```

### Expected Runtime
- **CPU:** ~10-15 minutes
- **GPU:** ~5-8 minutes

### Output
The notebook generates:
- ✓ 9 regression model results
- ✓ 5 classification model results
- ✓ 23+ visualizations
- ✓ Learning curves, confusion matrices, ROC curves
- ✓ Feature importance rankings
- ✓ Cross-validation analysis
- ✓ Comprehensive results tables

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── README.md                                    # This file
├── LICENSE                                      # MIT License
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore rules
├── Educational_Performance_ML_DL_Pipeline.ipynb # Main analysis (11 experiments)
├── data/                                        # Data information
│   └── README.md                                # Data source & download instructions
```

---

## 🔬 Methodology

### Approach

**Stage 1: Data Preparation**
- Load 14,003 student records from Zenodo
- Handle missing values and duplicates
- Engineer 4 domain-driven features
- Encode categorical variables
- Apply StandardScaler (ML) and MinMaxScaler (DL)
- 80-20 train-test split with random_state=42

**Stage 2: Traditional ML Models (6 experiments)**
- Linear Regression (baseline)
- Random Forest with GridSearchCV
- Gradient Boosting
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost Classifier

**Stage 3: Deep Learning Models (5 experiments)**
- Sequential API: Simple & Advanced architectures
- Functional API: Multi-path network
- Classification variants with proper regularization
- EarlyStopping and ReduceLROnPlateau callbacks
- L2 regularization, Batch Normalization, Dropout

**Stage 4: Evaluation**
- Regression: R², MAE, MSE, train-test gap
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC
- 5-fold cross-validation for stability
- Learning curves for overfitting diagnosis
- Confusion matrices for error analysis

---

## 🔄 Reproducibility

**Full Reproducibility Guaranteed:**
- ✅ Random seed set to 42 (NumPy & TensorFlow)
- ✅ Dataset sourced from public Zenodo URL
- ✅ All preprocessing documented
- ✅ Hyperparameters explicitly listed
- ✅ No hardcoded paths or dependencies
- ✅ Notebook runs top-to-bottom without errors
- ✅ Runtime: ~10-15 minutes on standard CPU

To reproduce:
```bash
jupyter notebook Educational_Performance_ML_DL_Pipeline.ipynb
# Run all cells (Ctrl+A, then Ctrl+Enter)
# Exact results will be generated
```

---

## 🛠️ Requirements

See `requirements.txt` for complete list. Key packages:

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.13.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** [Zenodo Contributors](https://zenodo.org/records/16459132) - Student Performance Dataset
- **Libraries:** TensorFlow, Scikit-learn, Pandas, Matplotlib, Seaborn
- **Inspiration:** Educational equity and data-driven intervention design
- **Project Mission:** Bridging education gaps for rural students

---

**Last Updated:** 19th October 2025  
**Reproducibility:** ✅ Fully Reproducible (seed=42)

---

## ⭐ If This Helped You

Please consider:
- ⭐ Starring this repository
- 🔗 Linking to this project
- 📝 Citing in your work
- 🤝 Contributing improvements

---

**Built with ❤️ for Educational Equity**