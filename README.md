# Diabetes Risk Prediction Using Machine Learning

BITSF464 Machine Learning – Phase 2 code submission
Sherwin Gummadi · Sreya Manoj · Shaikh Abdul Faheem

This project predicts diabetes risk from clinical measurements. It compares Logistic Regression, Random Forest,
Gradient Boosting and XGBoost under 5-fold stratified cross-validation, with and without SMOTE, explains the best
model with SHAP, and includes a small Streamlit prototype (risk calculator plus an experimental face-scan step).

> **Important:** the dataset is **synthetic** (not real patient data). The results show that the pipeline works;
> they do not establish clinical validity. The face-scan step of the app is an unvalidated prototype feature.

---

## 1. Contents

| File | What it is |
|---|---|
| `notebooks/01_EDA_and_initial_models.ipynb` | Data checks, exploratory data analysis, preprocessing, feature engineering, first model comparison, hold-out test, subgroup check and SHAP |
| `notebooks/02_results_for_article_with_outputs.ipynb` | The notebook that produced the tables and figures in the research article (saved **with** its outputs) |
| `app.py` | Streamlit web app (Step 1: clinical measurements → risk; Step 2: face scan) |
| `synthetic_diabetes_dataset.xlsx` | Dataset: 500 synthetic patients. Sheet **Patient Data** holds the data, sheet **Notes** describes the columns |
| `requirements.txt` | Python packages for the web app |
| `requirements-notebooks.txt` | Python packages for running the notebooks on your own computer (Colab installs what it needs) |

### Dataset
- 500 patients, 186 diabetic (37.2%) and 314 non-diabetic (62.8%).
- Columns: `Patient_ID`, `Age`, `Height_cm`, `Weight_kg`, `BMI`, `Blood_Pressure` (diastolic), `Insulin`,
  `Diabetes_Pedigree_Function`, `Outcome` (1 = diabetes, 0 = no diabetes).
- The file is small and is included, so no download is needed.

---

## 2. Dependencies

- Python 3.9 or newer.
- **Web app** (`requirements.txt`): streamlit, pandas, numpy, scikit-learn, openpyxl, streamlit-webrtc and
  `opencv-python-headless` **below version 5**.
  OpenCV 5 does not ship the face-detector files, so the face scan does not work with it. The requirements file
  already pins a version below 5.
- **Notebooks** (`requirements-notebooks.txt`): pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl, xgboost,
  imbalanced-learn (SMOTE), shap, jupyter.

---

## 3. How to run

### A. Notebooks in Google Colab (recommended)
1. Go to <https://colab.research.google.com> → **File → Upload notebook**, and choose a notebook from `notebooks/`.
2. Choose **Runtime → Run all**.
3. When asked, upload `synthetic_diabetes_dataset.xlsx`. The first cell installs `imbalanced-learn` and `shap`.
4. Run time is a few minutes (Random Forest and XGBoost run 8 configurations × 5 folds).

### B. Notebooks on your own computer
```
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
python3 -m pip install -r requirements-notebooks.txt
jupyter notebook
```
Open a notebook from `notebooks/`. Keep `synthetic_diabetes_dataset.xlsx` in the folder you start Jupyter from
(or upload it when the notebook asks).

### C. Streamlit web app
Open a terminal in the project folder (the one containing `app.py`) and run:
```
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```
The app opens at `http://localhost:8501`. Use `localhost`, because the camera only works there or over HTTPS.
Stop it with `Ctrl + C`.

**Using the app**
1. **Step 1:** enter age, height, weight, diastolic blood pressure, insulin and the diabetes pedigree function
   (use +/- for 0.5 steps) and click *Save measurements & calculate BMI*. BMI is calculated from height and weight.
2. **Step 2** unlocks only after Step 1 is saved. Choose *Live camera* (the green boxes follow your face) or
   *Photo / upload*, then click *Analyse*.
3. **Demo mode** (sidebar): loads a low-, moderate- or high-risk example patient and can use a simulated face
   scan, so the app can be shown without typing values or using a camera.

**Troubleshooting**
- *"The face detector is not available"* or *"no attribute CascadeClassifier"*:
  ```
  python3 -m pip uninstall -y opencv-python opencv-python-headless
  python3 -m pip install "opencv-python-headless<5"
  ```
- *"No face detected"*: keep the whole face inside the picture, face the camera, use even light, no glasses.
- *Live mode missing*: `python3 -m pip install streamlit-webrtc`, then restart.
- *`externally-managed-environment` error*: use the virtual environment steps above.

---

## 4. Expected outputs

### Notebooks (results from the saved run in notebook 02)
- **Cross-validated comparison** (5-fold stratified, mean ± std): Logistic Regression had the best ROC-AUC,
  **0.802 ± 0.035** (recall 0.570). Random Forest, Gradient Boosting and XGBoost scored 0.771–0.784, within
  fold-to-fold noise. Exact numbers vary slightly between runs and library versions.
- **Effect of SMOTE:** recall rose by about 0.08–0.14 for every model; ROC-AUC did not improve (change −0.001 to −0.010).
- **Hold-out test** (20%, 100 patients) of the selected model: accuracy 0.800, precision 0.774, recall 0.649,
  F1 0.706, ROC-AUC 0.833, with a confusion matrix and ROC curve.
- **Subgroup table** by age group and BMI group (recall is low for patients under 40 and with normal BMI).
- **Plots:** model comparison bars, confusion matrix, ROC curve, permutation importance, SHAP summary and
  SHAP waterfall (BMI is the most influential feature).
- **k check:** results for k = 3, 5 and 10.
- Notebook 02 also saves the figures as `fig4_cv_comparison.png`, `fig5_holdout.png`, `fig6_permutation.png`,
  `fig7a_shap_summary.png` and `fig7b_shap_waterfall.png`.

### Web app
- Sidebar: step progress and the model's cross-validated ROC-AUC (about 0.80).
- After Step 2: a combined **risk percentage** with a Low / Moderate / High label, the clinical-model part, the
  face-scan adjustment (at most ±5 points) and a table of the top factors behind the clinical risk.
- Demo patients (clinical model only): about 4% (low), 45% (moderate) and 97% (high).
- A warning that the output is **not a medical diagnosis**. Photos are processed in memory and are not saved.

---

## 5. Limitations
- Synthetic, small dataset (500 patients), mild class imbalance, no external validation.
- Hyperparameters were fixed and not tuned; the hold-out set is small (100 patients).
- The face-scan indicators are simple image heuristics with no supporting data in this project.
