'''
File for running churn_library functions
'''
# %%
from churn_library import (import_data, perform_eda, encoder_helper)


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

encoded_df = encoder_helper(data_df, category_lst)
print(encoded_df)

# %% perform_feature_engineering

# %% classification_report_image

# %% feature_importance_plot

# %% train_models