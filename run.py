'''
File for running churn_library functions
'''
# %%
from churn_library import import_data, perform_eda


# %% Import data
data_df = import_data('./data/bank_data.csv')
print(data_df)

# %% Perform EDA
perform_eda(data_df)




# %% encoder_helper

# %% perform_feature_engineering

# %% classification_report_image

# %% feature_importance_plot

# %% train_models