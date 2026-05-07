# 🏥 Health Insurance Premium Prediction

A modular Machine Learning pipeline designed to predict medical insurance costs based on individual health and demographic factors using **Linear Regression**.

## 🏗 Project Architecture
The project is built with a modular structure to ensure clean code and maintainability:
* **`src/data_loader.py`**: Handles data ingestion.
* **`src/preprocessing.py`**: Manages categorical encoding and data splitting.
* **`src/visualizer.py`**: Generates EDA plots (Age, Smoking impact, Correlation).
* **`src/model_trainer.py`**: Handles model training and serialization (.pkl).
* **`main.py`**: The main orchestrator of the pipeline.

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn, Joblib.

## 🚀 How to Run
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the project:
   `python main.py`

---
**Developed by:** [Maryam Lraimian]