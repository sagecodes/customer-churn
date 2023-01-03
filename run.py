'''
File for running churn_library functions
'''
# %%
from churn_library import (import_data,
                            perform_eda,
                            encoder_helper,
                            perform_feature_engineering,
                            classification_report_image,
                            feature_importance_plot)


# Temp imports
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

import pickle
import joblib 

# %% Import data
data_df = import_data('./data/bank_data.csv')
print(data_df)

# %% Perform EDA
perform_eda(data_df)




# %% encoder_helper

category_lst = ['Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category']

encoded_df = encoder_helper(data_df, category_lst, 'Churn')
print(encoded_df)

# %% perform_feature_engineering

target = 'Churn'

keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

print('Running perform_feature_engineering function:')
X_train, X_test, y_train, y_test = perform_feature_engineering(encoded_df, keep_cols, target)

#%%
# X = train_data
# y = data_df[target]
# y_train

#%%
# %% Temp training
# This cell may take up to 15-20 minutes to run
# train test split 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

# grid search
rfc = RandomForestClassifier(random_state=42)
# Use a different solver if the default 'lbfgs' fails to converge
# Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}

# cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
# cv_rfc.fit(X_train, y_train)

# lrc.fit(X_train, y_train)

cv_rfc = joblib.load('models\\rfc_model.pkl')
lrc = joblib.load('models\logistic_model.pkl')

# cv_rfc = pickle.load(open('models\rfc_model.pkl', 'rb'))
# lrc = pickle.load(open('models\logistic_model.pkl', 'rb'))

y_train_preds_rf = cv_rfc.predict(X_train)
y_test_preds_rf = cv_rfc.predict(X_test)

y_train_preds_lr = lrc.predict(X_train)
y_test_preds_lr = lrc.predict(X_test)

#%%

classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

# %% feature_importance_plot
# feature imoprtance for random forest
feature_importance_plot(cv_rfc, X_test,'images/random_forest_feature_importance.png')
# feature_importance_plot(lrc, X_test,'images/logistic_regression_feature_importance.png')

# %% train_models