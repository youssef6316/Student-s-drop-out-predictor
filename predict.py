import joblib
import numpy as np
import pandas as pd

# Load saved model and preprocessors
model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
nominal_encoder = joblib.load("nominal_encoder.pkl")
ordinal_encoder = joblib.load("ordinal_encoder.pkl")

# Define feature types
numerical_cols = [
                'Previous qualification (grade)',
                'Admission grade',
                'Age at enrollment',
                "Curricular units 1st sem (credited)",
                "Curricular units 1st sem (enrolled)",
                "Curricular units 1st sem (evaluations)",
                "Curricular units 1st sem (approved)",
                "Curricular units 1st sem (grade)",
                "Curricular units 1st sem (without evaluations)",

                "Curricular units 2nd sem (credited)",
                "Curricular units 2nd sem (enrolled)",
                "Curricular units 2nd sem (evaluations)",
                "Curricular units 2nd sem (approved)",
                "Curricular units 2nd sem (grade)",
                "Curricular units 2nd sem (without evaluations)",

                "GDP",
                "Inflation rate",

                "grade_increase_ratio",
                "grade_per_credit_sem_1",
                "grade_per_credit_sem_2",
                "grade_diff",
                "cumulative_approval_rate",
                "grade_points_avg",
                "age_penalty",
                "approved_rate_sem_1",
                "approved_rate_sem_2"
]
ordinal_cols = ['Application order']
nominal_cols = [
                'Marital status',
                'Course',
                'Daytime/evening attendance',
                'Previous qualification',
                'Nacionality',
                "Mother's qualification",
                "Father's qualification",
                "Mother's occupation",
                "Father's occupation",
                'Displaced',
                "Educational special needs",
                "Debtor",
                "Tuition fees up to date",
                'Gender',
                'Scholarship holder',
                'improved'
]

def predict_from_user_input(user_input: dict):
    df = pd.DataFrame([user_input])  # Turn dict into 1-row DataFrame

    removed_columns = [
        "Course_Animation and Multimedia Design",
        "Previous qualification_10th year of schooling",
        "Previous qualification_10th year of schooling - not completed",
        "Previous qualification_doctorate",
        "Nacionality_English",
        "Nacionality_Mozambican",
        "Mother's qualification_2nd cycle general HS",
        "Mother's qualification_Can't read/write",
        "Mother's qualification_Doctorate (3rd cycle)",
        "Father's qualification_Admin & Commerce",
        "Father's qualification_Can't read/write",
        "Father's qualification_Complementary HS Not Concluded",
        "Father's qualification_Doctorate (3rd cycle)",
        "Father's qualification_Specialized higher",
        "Mother's occupation_Armed Forces",
        "Mother's occupation_Finance/statistics/registry ops",
        "Mother's occupation_ICT specialists",
        "Mother's occupation_Legal/social/sports techs",
        "Father's occupation_Admin service managers",
        "Father's occupation_Army Officers",
        "Father's occupation_Army Sergeants",
        "Father's occupation_Finance/PR/Org specialists",
        "Father's occupation_Finance/statistics/registry ops",
        "Father's occupation_ICT techs",
        "Father's occupation_Metal workers",
        "Father's occupation_Secretaries & data ops"
    ]

    # Apply encoders
    encoded_nominal = nominal_encoder.transform(df[nominal_cols])

    encoded_ordinal = ordinal_encoder.transform(df[ordinal_cols])

    scaled_numerical = scaler.transform(np.hstack([df[numerical_cols], encoded_ordinal]))

    # Get the encoded feature names
    all_columns = nominal_encoder.get_feature_names_out()

    # Get indices of the columns to keep
    keep_indices = [i for i, col in enumerate(all_columns) if col not in removed_columns]

    # Drop the columns
    encoded_nominal_cleaned = encoded_nominal[:, keep_indices]

    # Concatenate all features into final model input
    X_input = np.hstack([scaled_numerical, encoded_nominal_cleaned])

    # Predict
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0]

    return prediction, probability

# EXAMPLE USAGE
# if __name__ == "__main__":
user_data = {
    "Marital status": "single",
    "Application order": "4th choice",
    "Course": "Social Service",
    "Daytime/evening attendance": "daytime",

    "Previous qualification": "Secondary education",
    "Previous qualification (grade)": 137.0,

    "Nacionality": "Romanian",

    "Mother's qualification": "Secondary Education",
    "Father's qualification": "Secondary Education",
    "Mother's occupation": "Unskilled Workers",
    "Father's occupation": "Unskilled Workers",

    "Admission grade": 129.3,
    "Displaced": "no",
    "Educational special needs": "no",
    "Debtor": "no",
    "Tuition fees up to date": "yes",
    "Gender": "Female",
    "Scholarship holder": "yes",
    "Age at enrollment": 21,

    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 8,
    "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 13.875,
    "Curricular units 1st sem (without evaluations)": 0,

    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 7,
    "Curricular units 2nd sem (approved)": 6,
    "Curricular units 2nd sem (grade)": 14.14,
    "Curricular units 2nd sem (without evaluations)": 0,

    "GDP": -0.92,
    "Inflation rate": 0.3,
}

# features of "Curricular Units 1st Semester" will be used to extract new features and dropping some to increase quality of data
#
#  increase ratio in grade = (sem_1 grade + sem_2 grade) / admission grade
#  grade per credit = grade / approved
#  grade diff = grades difference (2nd vs 1st)
#  improved (bool depends on grade_diff)
#  cum approval rate = (approve + approve)/(eval+eval)
#  gpa = ((evaluated credits sem 1 * sem 1 grade)+
#         (evaluated credits of sem 2 * sem 2 grade)) /
#         sum of evaluated credits of both semesters
#  age penalty = 1 / (enrollment age +1)
#  approved rate = approved /Evaluated

added_features = {
    "grade_increase_ratio": (user_data["Curricular units 1st sem (grade)"] +
                             user_data["Curricular units 2nd sem (grade)"]) /
                             user_data["Admission grade"],

    "grade_per_credit_sem_1": user_data['Curricular units 1st sem (grade)'] /
                              (user_data['Curricular units 1st sem (approved)'] + 1e-5),
    "grade_per_credit_sem_2": user_data['Curricular units 2nd sem (grade)'] /
                              (user_data['Curricular units 2nd sem (approved)'] + 1e-5),
    "grade_diff": user_data['Curricular units 2nd sem (grade)'] - user_data['Curricular units 1st sem (grade)'],
    "improved": 1 if (user_data['Curricular units 2nd sem (grade)']
                 - user_data['Curricular units 1st sem (grade)'] > 0) else 0,
    "cumulative_approval_rate": (user_data['Curricular units 1st sem (approved)'] + 
                                 user_data['Curricular units 2nd sem (approved)']) 
                                / (user_data['Curricular units 1st sem (evaluations)'] 
                                + user_data['Curricular units 2nd sem (evaluations)'] + 1e-5),
    "grade_points_avg": (
    user_data['Curricular units 1st sem (grade)'] * user_data['Curricular units 1st sem (evaluations)'] +
    user_data['Curricular units 2nd sem (grade)'] * user_data['Curricular units 2nd sem (evaluations)']
) / (
    user_data['Curricular units 1st sem (evaluations)'] + user_data['Curricular units 2nd sem (evaluations)'] + 1e-5
),
    "age_penalty": 1 / (user_data['Age at enrollment'] + 1),
    "approved_rate_sem_1": user_data['Curricular units 1st sem (approved)'] / 
                           (user_data['Curricular units 1st sem (evaluations)'] + 1e-5),
    "approved_rate_sem_2": user_data['Curricular units 2nd sem (approved)'] / (user_data['Curricular units 2nd sem (evaluations)'] + 1e-5)
}

user_data.update(added_features)

prediction_value, prob = predict_from_user_input(user_data)
prediction = "Drop Out" if prediction_value == 0 else "Graduate"
print(f"Predicted class: {prediction}")
print(f"Probabilities: {prob[0]}, {prob[1]}")
