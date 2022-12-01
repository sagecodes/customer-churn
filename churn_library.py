# library doc string


# import libraries
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(pth):
   '''
   returns dataframe for the csv found at pth

   input:
        pth: a path to the csv
   output:
        df: pandas dataframe
   '''	
   df = pd.read_csv(pth)

   return df


def perform_eda(df):
   '''
   perform eda on df and save figures to images folder
   input:
        df: pandas dataframe

   output:
        None
   '''

   print('--- EDA ---')
   print('Shape:')
   print(df.shape)
   print('\nNull Values:')
   print(df.isnull().sum())
   print('\nDescribe')
   print(df.describe())


   cat_columns = [
       'Gender',
       'Education_Level',
       'Marital_Status',
       'Income_Category',
       'Card_Category'                
   ]

   quant_columns = [
       'Customer_Age',
       'Dependent_count', 
       'Months_on_book',
       'Total_Relationship_Count', 
       'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 
       'Credit_Limit', 
       'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 
       'Total_Amt_Chng_Q4_Q1', 
       'Total_Trans_Amt',
       'Total_Trans_Ct', 
       'Total_Ct_Chng_Q4_Q1', 
       'Avg_Utilization_Ratio'
   ]

   # Save graph: Attrition_Flag for Existing Customer
   df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
   fig = plt.figure(figsize=(20,10)) 
   df['Churn'].hist();
   plt.savefig('images/Attrition_Flag_Existing_Customer.png')
   plt.close(fig)

   # 
   plt.figure(figsize=(20,10)) 
   df['Customer_Age'].hist();

   plt.figure(figsize=(20,10)) 
   df.Marital_Status.value_counts('normalize').plot(kind='bar');

   plt.figure(figsize=(20,10)) 
   # distplot is deprecated. Use histplot instead
   # sns.distplot(df['Total_Trans_Ct']);
   # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
   sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True);

   plt.figure(figsize=(20,10)) 
   sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
   plt.show()



# def encoder_helper(df, category_lst, response):
#     '''
#     helper function to turn each categorical column into a new column with
#     propotion of churn for each category - associated with cell 15 from the notebook

#     input:
#             df: pandas dataframe
#             category_lst: list of columns that contain categorical features
#             response: string of response name [optional argument that could be used for naming variables or index y column]

#     output:
#             df: pandas dataframe with new columns for
#     '''
#     pass


# def perform_feature_engineering(df, response):
#     '''
#     input:
#               df: pandas dataframe
#               response: string of response name [optional argument that could be used for naming variables or index y column]

#     output:
#               X_train: X training data
#               X_test: X testing data
#               y_train: y training data
#               y_test: y testing data
#     '''

# def classification_report_image(y_train,
#                                 y_test,
#                                 y_train_preds_lr,
#                                 y_train_preds_rf,
#                                 y_test_preds_lr,
#                                 y_test_preds_rf):
#     '''
#     produces classification report for training and testing results and stores report as image
#     in images folder
#     input:
#             y_train: training response values
#             y_test:  test response values
#             y_train_preds_lr: training predictions from logistic regression
#             y_train_preds_rf: training predictions from random forest
#             y_test_preds_lr: test predictions from logistic regression
#             y_test_preds_rf: test predictions from random forest

#     output:
#              None
#     '''
#     pass


# def feature_importance_plot(model, X_data, output_pth):
#     '''
#     creates and stores the feature importances in pth
#     input:
#             model: model object containing feature_importances_
#             X_data: pandas dataframe of X values
#             output_pth: path to store the figure

#     output:
#              None
#     '''
#     pass

# def train_models(X_train, X_test, y_train, y_test):
#     '''
#     train, store model results: images + scores, and store models
#     input:
#               X_train: X training data
#               X_test: X testing data
#               y_train: y training data
#               y_test: y testing data
#     output:
#               None
#     '''
#     pass