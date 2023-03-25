"""
File for running churn_library functions

Can be run in VS Code with the Python Interactive window "# %%"

or in the terminal with the command: python run.py

author: @sagecodes
date: 03/13/2023

"""
# %% Import libraries and churn library functions
import joblib


from churn_library import (
    encoder_helper,
    feature_importance_plot,
    import_data,
    perform_eda,
    perform_feature_engineering,
    train_models,
    save_roc_curve,
    shap_values_plot,
    classification_report_image,
)

# %% Import data
data_df = import_data("./data/bank_data.csv")
print(data_df)

# %% Perform EDA & save results to images folder
perform_eda(data_df)

# %% encoder_helper
category_lst = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

encoded_df = encoder_helper(data_df, category_lst, "Churn")
print(encoded_df)

# %% perform_feature_engineering
target = "Churn"

keep_cols = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]

print("Running perform_feature_engineering function:")
X_train, X_test, y_train, y_test = perform_feature_engineering(
    encoded_df, keep_cols, target
)


# %% train_models
train_models(X_train, X_test, y_train, y_test)

# %% load trained models
cv_rfc = joblib.load("models\\rfc_model_train.pkl")
cv_lr = joblib.load("models\\logistic_model_train.pkl")

# %% feature imoprtance for random forest
feature_importance_plot(cv_rfc, X_test, "images/random_forest_feature_importance.png")

# %% plot & save roc curves
save_roc_curve(cv_lr, cv_rfc, X_test, y_test)

# %% save shap values
shap_values_plot(cv_rfc, X_test)

