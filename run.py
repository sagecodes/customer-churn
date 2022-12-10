'''
File for running churn_library functions
'''
# %%
from churn_library import (import_data,
                            perform_eda,
                            encoder_helper,
                            perform_feature_engineering)


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


train_data = perform_feature_engineering(encoded_df, keep_cols, target)

#%%
train_data[2]

# %% classification_report_image

# %% feature_importance_plot

# %% train_models