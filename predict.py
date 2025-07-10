import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# Load model and encoders
model = joblib.load("model/mlp_model.pkl")
scaler = joblib.load("model/scaler.pkl")
nominal_encoder = joblib.load("model/nominal_encoder.pkl")
ordinal_encoder = joblib.load("model/ordinal_encoder.pkl")

# Define feature types
numerical_cols = [
    'Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "GDP", "Inflation rate", "grade_increase_ratio", "grade_per_credit_sem_1",
    "grade_per_credit_sem_2", "grade_diff", "cumulative_approval_rate",
    "grade_points_avg", "age_penalty", "approved_rate_sem_1", "approved_rate_sem_2"
]
ordinal_cols = ['Application order']
nominal_cols = [
    'Marital status', 'Course', 'Daytime/evening attendance', 'Previous qualification',
    'Nacionality', "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Displaced',
    "Educational special needs", "Debtor", "Tuition fees up to date",
    'Gender', 'Scholarship holder', 'improved'
]

removed_columns = [  # Columns to be dropped from nominal encoder
    "Course_Animation and Multimedia Design", "Previous qualification_10th year of schooling",
    "Previous qualification_10th year of schooling - not completed", "Previous qualification_doctorate",
    "Nacionality_English", "Nacionality_Mozambican", "Mother's qualification_2nd cycle general HS",
    "Mother's qualification_Can't read/write", "Mother's qualification_Doctorate (3rd cycle)",
    "Father's qualification_Admin & Commerce", "Father's qualification_Can't read/write",
    "Father's qualification_Complementary HS Not Concluded", "Father's qualification_Doctorate (3rd cycle)",
    "Father's qualification_Specialized higher", "Mother's occupation_Armed Forces",
    "Mother's occupation_Finance/statistics/registry ops", "Mother's occupation_ICT specialists",
    "Mother's occupation_Legal/social/sports techs", "Father's occupation_Admin service managers",
    "Father's occupation_Army Officers", "Father's occupation_Army Sergeants",
    "Father's occupation_Finance/PR/Org specialists", "Father's occupation_Finance/statistics/registry ops",
    "Father's occupation_ICT techs", "Father's occupation_Metal workers",
    "Father's occupation_Secretaries & data ops"
]

# === INPUT HELPERS ===

def get_input(prompt, cast_type=str):
    while True:
        try:
            return cast_type(input(prompt))
        except ValueError:
            print("Invalid input. Please try again.")

def collect_user_input():
    dataset = pd.read_csv("Data/data.csv")

    def print_unique_values(col):
        if col in dataset.columns:
            values = dataset[col].dropna().unique()
            print(f"  [Options for '{col}']: {', '.join(map(str, values))}")

    user_data = {}

    # Categorical Inputs (with unique values)
    categorical_inputs = [
        "Marital status", "Application order", "Course", "Daytime/evening attendance",
        "Previous qualification", "Nacionality", "Mother's qualification",
        "Father's qualification", "Mother's occupation", "Father's occupation",
        "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date",
        "Gender", "Scholarship holder"
    ]
    for col in categorical_inputs:
        print_unique_values(col)
        user_data[col] = get_input(f"{col}: ")

    # Numerical Inputs with Range Hints
    numeric_inputs = {
        "Previous qualification (grade)": float,
        "Admission grade": float,
        "Age at enrollment": int,
        "Curricular units 1st sem (credited)": int,
        "Curricular units 1st sem (enrolled)": int,
        "Curricular units 1st sem (evaluations)": int,
        "Curricular units 1st sem (approved)": int,
        "Curricular units 1st sem (grade)": float,
        "Curricular units 1st sem (without evaluations)": int,
        "Curricular units 2nd sem (credited)": int,
        "Curricular units 2nd sem (enrolled)": int,
        "Curricular units 2nd sem (evaluations)": int,
        "Curricular units 2nd sem (approved)": int,
        "Curricular units 2nd sem (grade)": float,
        "Curricular units 2nd sem (without evaluations)": int,
        "GDP": float,
        "Inflation rate": float
    }
    grade_ranges = {
        "Previous qualification (grade)": "--/200",
        "Admission grade": "--/200",
        "Curricular units 1st sem (grade)": "--/20",
        "Curricular units 2nd sem (grade)": "--/20"
    }
    for col, typ in numeric_inputs.items():
        if col in grade_ranges:
            print(f"  [Hint for '{col}']: expected range {grade_ranges[col]}")
        user_data[col] = get_input(f"{col}: ", typ)

    # Derived features (added_features)
    user_data.update({
        "grade_increase_ratio": (user_data["Curricular units 1st sem (grade)"] +
                                 user_data["Curricular units 2nd sem (grade)"]) /
                                 user_data["Admission grade"],
        "grade_per_credit_sem_1": user_data['Curricular units 1st sem (grade)'] /
                                  (user_data['Curricular units 1st sem (approved)'] + 1e-5),
        "grade_per_credit_sem_2": user_data['Curricular units 2nd sem (grade)'] /
                                  (user_data['Curricular units 2nd sem (approved)'] + 1e-5),
        "grade_diff": user_data['Curricular units 2nd sem (grade)'] -
                      user_data['Curricular units 1st sem (grade)'],
        "improved": 1 if user_data['Curricular units 2nd sem (grade)'] >
                         user_data['Curricular units 1st sem (grade)'] else 0,
        "cumulative_approval_rate": (user_data['Curricular units 1st sem (approved)'] +
                                     user_data['Curricular units 2nd sem (approved)']) /
                                    (user_data['Curricular units 1st sem (evaluations)'] +
                                     user_data['Curricular units 2nd sem (evaluations)'] + 1e-5),
        "grade_points_avg": (
            user_data['Curricular units 1st sem (grade)'] *
            user_data['Curricular units 1st sem (evaluations)'] +
            user_data['Curricular units 2nd sem (grade)'] *
            user_data['Curricular units 2nd sem (evaluations)']) /
            (user_data['Curricular units 1st sem (evaluations)'] +
             user_data['Curricular units 2nd sem (evaluations)'] + 1e-5),
        "age_penalty": 1 / (user_data['Age at enrollment'] + 1),
        "approved_rate_sem_1": user_data['Curricular units 1st sem (approved)'] /
                               (user_data['Curricular units 1st sem (evaluations)'] + 1e-5),
        "approved_rate_sem_2": user_data['Curricular units 2nd sem (approved)'] /
                               (user_data['Curricular units 2nd sem (evaluations)'] + 1e-5)
    })

    return user_data

# === PREDICTION ===

def predict_from_user_input(user_input: dict):
    df = pd.DataFrame([user_input])
    encoded_nominal = nominal_encoder.transform(df[nominal_cols])
    encoded_ordinal = ordinal_encoder.transform(df[ordinal_cols])
    scaled_numerical = scaler.transform(np.hstack([df[numerical_cols], encoded_ordinal]))

    all_columns = nominal_encoder.get_feature_names_out()
    keep_indices = [i for i, col in enumerate(all_columns) if col not in removed_columns]
    encoded_nominal_cleaned = encoded_nominal[:, keep_indices]

    X_input = np.hstack([scaled_numerical, encoded_nominal_cleaned])
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0]
    return prediction, probability

# === MAIN ===

if __name__ == "__main__":
    print("===== STUDENT OUTCOME PREDICTOR =====\n")
    user_data = collect_user_input()
    prediction_value, prob = predict_from_user_input(user_data)
    prediction_label = "Drop Out" if prediction_value == 0 else "Graduate"

    print("\n===== PREDICTION RESULT =====")
    print(f"Predicted Class: {prediction_label}")
    print(f"Probability - Drop Out: {prob[0]*100:.4f }%, Graduate: {prob[1]*100:.4f}%")
