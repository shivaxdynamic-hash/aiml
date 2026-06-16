# aiml
hello this is my first MACHINE LEARNING project
# 🚢 Titanic Survival Prediction

> A complete Machine Learning pipeline to predict passenger survival on the Titanic — covering data cleaning, preprocessing, feature engineering, model training, and evaluation using Python and Scikit-learn.

---

## 📌 Project Overview

This project uses the classic **Titanic dataset** to build and compare multiple supervised machine learning classification models. The goal is to predict whether a passenger survived the Titanic disaster based on features like age, gender, passenger class, fare, and embarkation point.

The project demonstrates a full end-to-end ML workflow — from raw data to model evaluation — built in **Google Colab** using Python.

---

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── TITANIC.ipynb        # Main Colab notebook (full pipeline)
├── titanic.csv          # Dataset (891 passengers, 12 features)
└── README.md            # Project documentation
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | Titanic passenger records |
| Total Records | 891 passengers |
| Features | 12 columns |
| Target Variable | `Survived` (0 = No, 1 = Yes) |
| Survival Rate | ~38.4% (342 out of 891) |

### Feature Description

| Column | Type | Description |
|---|---|---|
| `PassengerId` | int | Unique passenger identifier |
| `Survived` | int | Target — 0 = No, 1 = Yes |
| `Pclass` | int | Ticket class (1st, 2nd, 3rd) |
| `Name` | object | Passenger name |
| `Sex` | object | Gender |
| `Age` | float | Age in years |
| `SibSp` | int | No. of siblings/spouses aboard |
| `Parch` | int | No. of parents/children aboard |
| `Ticket` | object | Ticket number |
| `Fare` | float | Passenger fare |
| `Cabin` | object | Cabin number (many missing) |
| `Embarked` | object | Port of embarkation (C, Q, S) |

---

## ⚙️ ML Pipeline

### Step 1 — Data Loading & Exploration
- Loaded dataset using `pd.read_csv()`
- Inspected shape, data types, and basic statistics via `head()`, `info()`, `describe()`

### Step 2 — Handling Missing Values

| Column | Missing Count | Action Taken |
|---|---|---|
| `Age` | 177 | Filled with **column mean** |
| `Cabin` | 687 | **Dropped** (too many nulls) |
| `Embarked` | 2 | **Rows dropped** |

After cleaning: **889 rows, 11 columns**, zero null values.

### Step 3 — Duplicate Check
- Result: **0 duplicate rows** confirmed

### Step 4 — Label Encoding
- Applied `LabelEncoder` to all categorical columns: `Name`, `Sex`, `Ticket`, `Embarked`
- Converted object types to numerical format for model compatibility

### Step 5 — Feature & Target Split
```python
X = df.drop(['Survived'], axis=1)   # Input features (10 columns)
Y = df['Survived']                   # Target variable
```

### Step 6 — Train-Test Split
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

| Set | Rows |
|---|---|
| Training set | 711 (80%) |
| Test set | 178 (20%) |

### Step 7 — Feature Scaling
- Applied `StandardScaler` to normalize all features
- `fit_transform()` on training set, `transform()` on test set

---

## 🤖 Models Trained & Compared

Five classification models were trained and evaluated:

| Model | Accuracy |
|---|---|
| Logistic Regression | 79.21% |
| Decision Tree | 78.65% |
| Random Forest | 80.90% |
| KNN | 78.65% |
| **SVM (Best)** ✅ | **82.02%** |

---

## 📈 Best Model — SVM Evaluation

The **Support Vector Machine (SVM)** achieved the highest accuracy of **82.02%**.

### Confusion Matrix
```
[[92  17]
 [15  54]]
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (Not Survived) | 0.86 | 0.84 | 0.85 | 109 |
| 1 (Survived) | 0.76 | 0.78 | 0.77 | 69 |
| **Accuracy** | | | **0.82** | **178** |
| Macro Avg | 0.81 | 0.81 | 0.81 | 178 |
| Weighted Avg | 0.82 | 0.82 | 0.82 | 178 |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3 | Programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML models, preprocessing, evaluation |
| SciPy | Statistical utilities |
| Google Colab | Development environment |

### Libraries & Modules
```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Open in Google Colab**
   - Upload `TITANIC.ipynb` to [colab.research.google.com](https://colab.research.google.com)
   - Upload `titanic.csv` to your Google Drive under `My Drive/`

3. **Run all cells** in order from top to bottom

---

## ✅ Key Takeaways

- Handled real-world messy data with missing values and categorical features
- Compared 5 different classification algorithms on the same dataset
- **SVM outperformed** all other models with **82.02% accuracy**
- Applied proper ML best practices: train/test split, feature scaling, label encoding

---

## 👨‍💻 Author

**Siva Kumar Thota**  
Aspiring Data Analyst | Aditya University  
📧 veerasivakumar143@gmail.com  
🔗 [GitHub Profile](https://github.com/your-username)

---

*Built with Python 🐍 | Google Colab ☁️ | Scikit-learn 🤖*
