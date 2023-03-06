import os
import logging
import churn_library as cls

from churn_library import (import_data,
							perform_eda,
							encoder_helper,
							)

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
	datefmt='%Y-%m-%d %H:%M:%S')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
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
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	'''
	test perform eda function
	'''
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
	'''
	test encoder helper
	'''
	df = import_data("./data/bank_data.csv")

	category_lst = ['Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category']
	target = 'Churn'

	encoded_df = encoder_helper(df, category_lst, target)

	for col in category_lst:
		try:
			assert col + "_" + target in encoded_df.columns
			logging.info("Testing encoder_helper: SUCCESS")
		except AssertionError as err:
			logging.error("Testing encoder_helper: missing column")
			raise err
		
	

def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	test_import()
	# df = import_data("./data/bank_data.csv")
	test_eda()
	test_encoder_helper(encoder_helper)








