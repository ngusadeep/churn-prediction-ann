# Customer Churn Prediction using ANN

This project demonstrates the use of a feed-forward Artificial Neural Network (ANN) to predict customer churn using the IBM Telco Customer Churn dataset. It covers data preprocessing, feature engineering, model training, evaluation, and visualization, providing a complete workflow for churn prediction.


## Overview & Objective
- **Objective**: Predict whether a customer will churn (binary classification).  
- **Target Variable**: `Churn` column.  
- **Approach**: 
  1. Encode categorical features and scale numerical features.  
  2. Split dataset into training and test sets.  
  3. Train a compact ANN using Keras/TensorFlow.  
  4. Evaluate performance using accuracy, confusion matrix, and classification metrics.


## Project Structure
- `prediction.ipynb` — main notebook containing the full workflow.  
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — dataset used.  
- `README.md` — project documentation.


## Dataset
- **Source**: IBM Telco Customer Churn dataset.  
- **Description**: Includes demographic, service, and billing information of customers, with a `Churn` label indicating whether the customer left the service.  



## Model Architecture
- **Input Features**: 26  
- **Layers**:
  ```python
  Dense(26, activation='relu', input_shape=(26,))
  Dense(1, activation='sigmoid')
  ```

* **Compilation**:

  * Optimizer: `adam`
  * Loss: `binary_crossentropy`
  * Metrics: `accuracy`
* **Training**: 100 epochs, validation split 0.2


## Preprocessing Steps

1. **Categorical Encoding**: One-hot encoding for features like `InternetService`, `Contract`, `PaymentMethod`.
2. **Numerical Scaling**: Min-Max scaling for `tenure`, `MonthlyCharges`, `TotalCharges`.
3. **Train-Test Split**: 80% training, 20% testing (`random_state=5`).
4. **Input Shape**: Ensured consistency with 26 features.



## Evaluation

* **Metrics Used**: Accuracy, Confusion Matrix, Precision, Recall, F1-score.
* **Performance**: Validation accuracy typically \~0.80.



## Environment & Requirements

* **Python 3.x**
* Install dependencies:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```



## How to Run

1. Install dependencies.
2. Open `prediction.ipynb` in Jupyter Lab, Notebook, or VS Code.
3. Run all cells sequentially to reproduce preprocessing, training, and evaluation.



## Notes

* Ensure `TotalCharges` is numeric before scaling.
* Random seed (`random_state=5`) ensures reproducibility.
* GPU recommended for faster model training.



## Acknowledgements

* IBM Telco Customer Churn dataset
* TensorFlow/Keras and scikit-learn communities

```

I can also create a **matching `requirements.txt` and GitHub repo description** that complements this README professionally. Do you want me to do that?
```
