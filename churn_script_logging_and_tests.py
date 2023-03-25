'''
this script is used to test the functions in the churn_library.py file

to run this script, use the command: python churn_script_logging_and_tests.py

author: @sagecodes
date: 03/25/2023
'''

import logging
import os

import joblib

from churn_library import (
    classification_report_image,
    encoder_helper,
    feature_importance_plot,
    import_data,
    perform_eda,
    perform_feature_engineering,
    train_models,
    save_roc_curve,
    shap_values_plot,
)

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Parameters used in test functions
###########################################################

category_lst = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

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

target = "Churn"

# test functions
###########################################################


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_eda():
    """
    test perform eda function
    """
    df = import_data("./data/bank_data.csv")
    try:
        perform_eda(df)
        assert os.path.exists("./images/Attrition_Flag_Existing_Customer.png")
        assert os.path.exists("./images/Customer_Age.png")
        assert os.path.exists("./images/Total_Trans_Ct.png")
        assert os.path.exists("./images/correlation.png")
        logging.info("Testing perform_eda - All images saved: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda - Images saved: file(s) not found")
        raise err


def test_encoder_helper(encoder_helper):
    """
    test encoder helper
    """
    df = import_data("./data/bank_data.csv")

    encoded_df = encoder_helper(df, category_lst, target)

    for col in category_lst:
        try:
            assert col + "_" + target in encoded_df.columns
            # logging.info("Testing encoder_helper: SUCCESS")
        except AssertionError as err:
            logging.error("Testing encoder_helper: missing column(s)")
            raise err
    logging.info("Testing encoder_helper: SUCCESS")


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """
    df = import_data("./data/bank_data.csv")

    encoded_df = encoder_helper(df, category_lst, target)

    # train_test_values = perform_feature_engineering(df)
    train_X, test_X, train_y, test_y = perform_feature_engineering(
        encoded_df, keep_cols, target
    )

    try:
        assert train_X.shape[0] > 0
        assert train_X.shape[1] > 0
        assert test_X.shape[0] > 0
        assert test_X.shape[1] > 0
        assert train_y.shape[0] > 0
        assert test_y.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: missing column(s) in train or test data"
        )
        raise err


def test_classification_report_image(classification_report_image):
    """
    test classification_report_image
    """
    df = import_data("./data/bank_data.csv")

    encoded_df = encoder_helper(df, category_lst, target)

    # train_test_values = perform_feature_engineering(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        encoded_df, keep_cols, target
    )

    cv_rfc = joblib.load("models\\rfc_model.pkl")
    lrc = joblib.load("models\\logistic_model.pkl")

    y_train_preds_rf = cv_rfc.predict(X_train)
    y_test_preds_rf = cv_rfc.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    try:
        assert os.path.exists("images\\logistic_regression_classification_report.png")
        assert os.path.exists("images\\random_forest_classification_report.png")
        logging.info("Testing classification_report_image - images saved: SUCCESS")
    except AssertionError as err:
        logging.error("Testing classification_report_image - image not saved")
        raise err


def test_feature_importance_plot(feature_importance_plot):
    """
    test feature_importance_plot
    """

    df = import_data("./data/bank_data.csv")

    encoded_df = encoder_helper(df, category_lst, target)

    # train_test_values = perform_feature_engineering(df)
    train_X, test_X, train_y, test_y = perform_feature_engineering(
        encoded_df, keep_cols, target
    )

    cv_rfc = joblib.load("models\\rfc_model.pkl")
    feature_importance_plot(
        cv_rfc, test_X, "images/random_forest_feature_importance.png"
    )

    try:
        assert os.path.exists("images/random_forest_feature_importance.png")
        logging.info("Testing feature_importance_plot - image saved: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot - Images saved: file(s) not found"
        )
        raise err
    
def test_save_roc_curve(save_roc_curve):
    """
    test save_roc_curve
    """
    df = import_data("./data/bank_data.csv")

    encoded_df = encoder_helper(df, category_lst, target)

    # train_test_values = perform_feature_engineering(df)
    train_X, test_X, train_y, test_y = perform_feature_engineering(
        encoded_df, keep_cols, target
    )

    rfc = joblib.load("models\\rfc_model.pkl")
    lrc = joblib.load("models\\logistic_model.pkl")

    save_roc_curve(lrc, rfc, test_X, test_y)

    try:
        assert os.path.exists("images\\roc_curve_lcr.png")
        assert os.path.exists("images\\random_forest_classification_report.png")
        assert os.path.exists("images\\roc_curve_rfc.png")
        logging.info("Testing save_roc_curve - images saved: SUCCESS")
    except AssertionError as err:
        logging.error("Testing save_roc_curve - image(s) not saved")
        raise err
    
def test_shap_values_plot(shap_values_plot):
    """
    test shap_values_plot
    """
    df = import_data("./data/bank_data.csv")

    encoded_df = encoder_helper(df, category_lst, target)

    # train_test_values = perform_feature_engineering(df)
    train_X, test_X, train_y, test_y = perform_feature_engineering(
        encoded_df, keep_cols, target
    )

    rfc = joblib.load("models\\rfc_model.pkl")

    shap_values_plot(rfc, test_X)

    try:
        assert os.path.exists("images\\shap_values.png")
        logging.info("Testing shap_values_plot - image saved: SUCCESS")
    except AssertionError as err:
        logging.error("Testing shap_values_plot - image not saved")
        raise err


def test_train_models(train_models):
    """
    test train_models
    """
    df = import_data("./data/bank_data.csv")

    encoded_df = encoder_helper(df, category_lst, target)

    train_X, test_X, train_y, test_y = perform_feature_engineering(
        encoded_df, keep_cols, target
    )

    train_models(train_X, test_X, train_y, test_y)

    try:
        assert os.path.exists("./models/logistic_model_train.pkl")
        assert os.path.exists("./models/rfc_model_train.pkl")
        logging.info("Testing train_models - All models saved: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models - Models saved: file(s) not found")
        raise err


# Run tests if script is called directly
if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_classification_report_image(classification_report_image)
    test_feature_importance_plot(feature_importance_plot)
    test_save_roc_curve(save_roc_curve)
    test_shap_values_plot(shap_values_plot)
    # test_train_models(train_models)
    print("Testing complete: see logs in /logs folder!")
