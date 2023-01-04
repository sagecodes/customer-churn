# library doc string


# import libraries
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import joblib 


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report


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


#    cat_columns = [
#        'Gender',
#        'Education_Level',
#        'Marital_Status',
#        'Income_Category',
#        'Card_Category'                
#    ]

#    quant_columns = [
#        'Customer_Age',
#        'Dependent_count', 
#        'Months_on_book',
#        'Total_Relationship_Count', 
#        'Months_Inactive_12_mon',
#        'Contacts_Count_12_mon', 
#        'Credit_Limit', 
#        'Total_Revolving_Bal',
#        'Avg_Open_To_Buy', 
#        'Total_Amt_Chng_Q4_Q1', 
#        'Total_Trans_Amt',
#        'Total_Trans_Ct', 
#        'Total_Ct_Chng_Q4_Q1', 
#        'Avg_Utilization_Ratio'
#    ]

   # Save graph: Attrition_Flag for Existing Customer

   df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
   fig = plt.figure(figsize=(20,10)) 
   df['Churn'].hist();
   plt.savefig('images/Attrition_Flag_Existing_Customer.png')
   plt.close(fig)
   print('saved Attrition graph at: images/Attrition_Flag_Existing_Customer.png')

   # save graph: Customer_Age
   plt.figure(figsize=(20,10)) 
   df['Customer_Age'].hist();
   plt.savefig('images/Customer_age.png')
   plt.close(fig)
   print('saved Customer Age graph at: images/Customer_Age.png')

   plt.figure(figsize=(20,10)) 
   df.Marital_Status.value_counts('normalize').plot(kind='bar');



   plt.figure(figsize=(20,10)) 
   # distplot is deprecated. Use histplot instead
   # sns.distplot(df['Total_Trans_Ct']);
   # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
   histplot = sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True);
   fig = histplot.get_figure()
   fig.savefig('images/Total_Trans_Ct.png')
   print('saved total transaction graph at: images/Total_Trans_Ct.png')


   # pairwise correlation
   plt.figure(figsize=(20,10)) 
   corr_plot = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
   fig = corr_plot.get_figure()
   fig.savefig('images/correlation.png')
   print('saved correlation graph at: images/correlation.png')


def encoder_helper(df, category_lst, target):
   '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            target: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for each category in list
   '''
   for col in category_lst:
      lst = []
      groups = df.groupby(col).mean()[target]

      for val in df[col]:
         lst.append(groups.loc[val])

      df[col+'_'+target] = lst
   return df
   


def perform_feature_engineering(df, keep_cols, target, test_size=0.2, seed=42):
    '''
    input:
              df: pandas dataframe
              target: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = df[keep_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
                                                      X, y,
                                                      test_size=test_size,
                                                      random_state=seed)

    return (X_train, X_test, y_train, y_test)



def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
   '''
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
   '''

   # Logistic regresison scores
   print('logistic regression results')
   print('test results')
   print(classification_report(y_test, y_test_preds_lr))
   print('train results')
   print(classification_report(y_train, y_train_preds_lr))

   # Random Forest scores
   print('random forest results')
   print('test results')
   print(classification_report(y_test, y_test_preds_rf))
   print('train results')
   print(classification_report(y_train, y_train_preds_rf))

   #create figure to save
      # fig = plt.figure(figsize=(20,10)) 

   plt.rc('figure', figsize=(6, 6))
   plt.text(0.01, 1.0, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
   plt.text(0.01, 0.65, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
   plt.text(0.01, 0.4, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
   plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
   plt.axis('off')
   plt.savefig('images/random_forest_classification_report.png')
   plt.close()

   plt.rc('figure', figsize=(6, 6))
   plt.text(0.01, 1.0, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
   plt.text(0.01, 0.65, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
   plt.text(0.01, 0.4, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
   plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
   plt.axis('off')
   plt.savefig('images/logistic_regression_classification_report.png')
   plt.close()
   

def feature_importance_plot(model, X_data, output_pth):
   '''
    creates and stores the feature importances in output_pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
   '''
   importances = model.feature_importances_
   indices = np.argsort(importances)[::-1]
   names = [X_data.columns[i] for i in indices]

   # Create plot
   plt.figure(figsize=(20,10))

   # Create plot title
   plt.title("Feature Importance")
   plt.ylabel('Importance')

   # Add bars
   plt.bar(range(X_data.shape[1]), importances[indices])

   # Add feature names as x-axis labels
   plt.xticks(range(X_data.shape[1]), names, rotation=90)
   plt.savefig(output_pth)
   plt.close()

   print(f'feature importance plot saved to {output_pth}')


def train_models(X_train, X_test, y_train, y_test):
   '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
   '''
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

   cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
   cv_rfc.fit(X_train, y_train)

   lrc.fit(X_train, y_train)

   y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
   y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

   y_train_preds_lr = lrc.predict(X_train)
   y_test_preds_lr = lrc.predict(X_test)

   # scores
   print('random forest results')
   print('test results')
   print(classification_report(y_test, y_test_preds_rf))
   print('train results')
   print(classification_report(y_train, y_train_preds_rf))

   print('logistic regression results')
   print('test results')
   print(classification_report(y_test, y_test_preds_lr))
   print('train results')
   print(classification_report(y_train, y_train_preds_lr))

   # save best model
   joblib.dump(cv_rfc.best_estimator_, './models/rfc_model_train.pkl')
   joblib.dump(lrc, './models/logistic_model_train.pkl')
   print('models saved!')