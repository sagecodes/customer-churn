"""
This library contains functions for the churn project

author: @sagecodes
date: 03/12/2023
"""

# import libraries
###########################################################
import os

import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

sns.set()


os.environ["QT_QPA_PLATFORM"] = "offscreen"


# Data: import & EDA
###########################################################
def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
         pth: a path to the csv
    output:
         df: pandas dataframe
    """
    df = pd.read_csv(pth)

    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    return df


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
         df: pandas dataframe

    output:
         None
    """
    print("--- EDA ---")
    print("Shape:")
    print(df.shape)
    print("\nNull Values:")
    print(df.isnull().sum())
    print("\nDescribe")
    print(df.describe())

    # save graph: Attrition_Flag
    fig = plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.savefig("images/Attrition_Flag_Existing_Customer.png")
    plt.close()
    print("saved Attrition graph at: images/Attrition_Flag_Existing_Customer.png")

    # save graph: Customer_Age
    fig = plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.savefig("images/Customer_age.png")
    plt.close()
    print("saved Customer Age graph at: images/Customer_Age.png")

    fig = plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts("normalize").plot(kind="bar")
    fig.savefig("images/Marital_Status.png")
    plt.close()
    print("saved total transaction graph at: images/Marital_Status.png")

    plt.figure(figsize=(20, 10))
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    histplot = sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    fig = histplot.get_figure()
    fig.savefig("images/Total_Trans_Ct.png")
    plt.close()
    print("saved total transaction graph at: images/Total_Trans_Ct.png")

    # pairwise correlation
    plt.figure(figsize=(20, 10))
    corr_plot = sns.heatmap(
        df.corr(),
        annot=False,
        cmap="Dark2_r",
        linewidths=2)
    fig = corr_plot.get_figure()
    fig.savefig("images/correlation.png")
    plt.close()
    print("saved correlation graph at: images/correlation.png")


# data: preprocessing & feature engineering
###########################################################
def encoder_helper(df, category_lst, target):
    """
    helper function to turn each categorical column into a new column with
      propotion of churn for each category - associated with cell 15 from
      the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            target: string of response name [optional argument that could be
               used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for each category in list
    """
    for col in category_lst:
        lst = []
        groups = df.groupby(col).mean()[target]

        for val in df[col]:
            lst.append(groups.loc[val])

        df[col + "_" + target] = lst
    return df


def perform_feature_engineering(df, keep_cols, target, test_size=0.2, seed=42):
    """
    input:
            df: pandas dataframe
            target: string of response name [optional argument that could be
               used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    X = df[keep_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    return (X_train, X_test, y_train, y_test)


# Models: Evaluation & Training
###########################################################
def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores
     report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """

    # Logistic regresison scores
    print("logistic regression results")
    print("test results")
    print(classification_report(y_test, y_test_preds_lr))
    print("train results")
    print(classification_report(y_train, y_train_preds_lr))

    # Random Forest scores
    print("random forest results")
    print("test results")
    print(classification_report(y_test, y_test_preds_rf))
    print("train results")
    print(classification_report(y_train, y_train_preds_rf))

    # create figure to save
    fig = plt.figure(figsize=(6,6))
    plt.text(
        0.01,
        1.0,
        str("Random Forest Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.65,
        str(classification_report(y_train, y_train_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.4,
        str("Random Forest Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig("images/random_forest_classification_report.png")
    plt.close()

    fig = plt.figure(figsize=(6,6))
    plt.text(
        0.01,
        1.0,
        str("Logistic Regression Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.65,
        str(classification_report(y_train, y_train_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.4,
        str("Logistic Regression Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig("images/logistic_regression_classification_report.png")
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in output_pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 10))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()

    print(f"feature importance plot saved to {output_pth}")


def save_roc_curve(lrc_model, rfc_model, X_test, y_test):
    """
    creates and stores the ROC curve in output_pth
    input:
            lrc_model: logistic regression model
            rfc_model: random forest model
            X_test: list of X test values
            y_test: list of y test values

    output:
            None
    """
    # save logistic regression roc curve
    fig = plt.figure(figsize=(6,6))
    lrc_plot = plot_roc_curve(lrc_model, X_test, y_test)
    plt.savefig("images/roc_curve_lcr.png")
    plt.close()

    # save random forest roc curve
    fig = plt.figure(figsize=(6,6))
    rfc_plot = plot_roc_curve(rfc_model, X_test, y_test)
    plt.savefig("images/roc_curve_rfv.png")
    plt.close()
    
    # save comparison of roc curves
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/roc_curve_comparison.png")
    plt.close()
    print("ROC comparison saved to: images/roc_curve_comparison.png")


def shap_values_plot(model, X_test):
    """
    creates and stores the shap values plot in output_pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    print("creating shap values plot this may take a while...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # fig = plt.figure(figsize=(15, 8))
    # shap.summary_plot(shap_values, X_test, plot_type="bar")
    fig = shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig("images/shap_values.png",dpi=300, bbox_inches='tight')
    plt.close()
    print("shap values plot saved to images/shap_values.png")


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    print("Training... This may take a while...")
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge

    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model_train.pkl")
    joblib.dump(lrc, "./models/logistic_model_train.pkl")
    print("models saved!")

    # save & print classification reports
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )
