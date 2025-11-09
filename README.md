# Customer Churn Prediction using Artificial Neural Network

## Overview
This project predicts whether a customer is likely to churn (leave the bank) based on their demographic and account details.  
The goal is to help financial institutions proactively identify at-risk customers and take timely retention measures.  

The model is built using an Artificial Neural Network (ANN) and deployed through a Streamlit application for real-time churn probability prediction.

---

## Project Workflow

### 1. Data Source
- Dataset: **Churn Modelling** (available on Kaggle)  
- [Dataset Link](https://www.kaggle.com/shrutimechlearn/churn-modelling)

The dataset contains 10,000 customer records with features such as geography, age, tenure, account balance, and activity details.

---

### 2. Data Preprocessing
- Checked for missing values and duplicates (none found).
- Converted categorical variables:
  - `Geography` → One-Hot Encoding  
  - `Gender` → Label Encoding
- Scaled numerical features using **StandardScaler** to ensure all features contribute equally to the model.
- Split the dataset into **80% training** and **20% testing** sets.

---

### 3. Feature Engineering
- Created dummy variables for the `Geography` column.  
- Verified feature relevance using **Lasso Regression** (L1 Regularization).  
- Final feature set included:
  ```
  ['Geo_France', 'Geo_Germany', 'Geo_Spain', 'CreditScore', 
   'Gender', 'Age', 'Tenure', 'Balance', 
   'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
  ```

---

### 4. Model Development

**Framework:** TensorFlow / Keras  
**Model Type:** Artificial Neural Network (ANN)

**Architecture:**
```python
model = Sequential()
model.add(Dense(6, activation='relu', input_shape=(12,)))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

**Compilation:**
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Training:**
- Batch Size: 32  
- Epochs: 40  

---

### 5. Model Performance

| Metric | Value |
|---------|-------|
| Test Accuracy | **85.7%** |
| Test Loss | **0.34** |

The ANN generalizes well without significant overfitting.  
A confusion matrix was also used to assess model performance and ensure balanced predictions between churn and non-churn classes.

---

### 6. Model Deployment

**Frontend:** Streamlit  
**Backend Model:** TensorFlow SavedModel format (Keras 3-compatible)  

- Trained model exported as a TensorFlow SavedModel (`model_export/`).
- Preprocessing scaler saved using **Joblib** (`scaler.pkl`).
- Streamlit web app built for user interaction and real-time predictions.

**Deployment Architecture:**
```
User Input (Streamlit Form)
        ↓
Feature Encoding + Scaling
        ↓
ANN Model Prediction (TensorFlow)
        ↓
Churn Probability & Classification Output
```

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Customer_Churn.git
cd Customer_Churn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

### 4. Access the App
Open the URL displayed in your terminal, typically:
```
http://localhost:8501
```

---

## Model Inputs and Outputs

### Input Features

| Feature | Type | Description |
|----------|------|-------------|
| Geography | Categorical | Country of customer (France, Spain, Germany) |
| Gender | Categorical | Male / Female |
| CreditScore | Numeric | Customer credit score (350–850) |
| Age | Numeric | Age of the customer |
| Tenure | Numeric | Years with the bank |
| Balance | Numeric | Current account balance |
| NumOfProducts | Numeric | Number of bank products used |
| HasCrCard | Binary | Has a credit card (1 = Yes, 0 = No) |
| IsActiveMember | Binary | Active customer status (1 = Yes, 0 = No) |
| EstimatedSalary | Numeric | Estimated annual salary |

### Output
- **Churn Probability:** Value between 0 and 1  
- **Prediction Class:**
  - `Stay` if probability ≤ 0.5  
  - `Churn` if probability > 0.5

---

## Key Learnings
- Handling **categorical encoding** correctly is crucial in ANN-based models.  
- **Feature scaling** ensures balanced learning in neural networks.  
- **Lasso regression** helps in feature importance interpretation even for non-linear models.  
- Learned how to save and deploy models using **TensorFlow SavedModel** and **Streamlit**.  
- Gained deeper understanding of **customer behavior analytics** and churn prediction use cases.

---

## Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| Language | Python |
| Framework | TensorFlow, Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib |
| Deployment | Streamlit |
| Serialization | Joblib, TensorFlow SavedModel |

---

## Business Impact
- Enables early identification of potential churners.  
- Supports customer retention campaigns by prioritizing high-risk customers.  
- Reduces marketing and acquisition costs through focused retention strategies.  
- Helps management make data-driven customer relationship decisions.

---

## Author
**Himanshu Sahu**  

