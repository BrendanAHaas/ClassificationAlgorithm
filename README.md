# ClassificationAlgorithm
Fully generalized ensemble machine learning algorithm written in python for filling in missing values in a dataset column.
This function aims to fill in the null/missing values in a column (thereby classifying those values)
This function does this by intaking a dataframe, looking at rows in the specified column with filled values, and using the information in other columns to determine how to fill in the missing values
Core file:  ClassificationAlgorithm.py
Helper file:  ClassificationUtils.py
Necessary Python packages:  pandas, numpy, matplotlib, scipy, sklearn

The concept for this project was to create an extremely generalized algorithm that could be used by individuals with no machine learning experience in order to fill in missing values in a column in their dataframe by using the rows without a missing value in that column as a training set for machine learning algorithms.  All the user has to do is to edit the variables data_folder, data_filename, and data_extension in ClassificationAlgorithm.py to indicate the correct data folder, and then enter which column they are trying to fill in missing values for.  Afterwards, they can run the file, and ClassificationAlgorithm.py will use an ensemble of machine learning algorithms to predict entries the missing values in the input document and will output the document with those missing entries now filled in with our predictions.  

This is primarily meant to be used as a way to impute missing values in a data column with machine learning techniques.  However, if a training data set is appended to a data set with values missing in the selected column, it can also serve as a standard machine learning predictor.

Outside of indicating the file and column to operate on, the rest of the process is fully automated.  Column types (numeric, categorical, datetime) can be manually input to expedite the process but can also be determined automatically.  Columns unlikely to be helpful in the machine learning process are automatically culled.  Numeric, categorical, and datetime columns can all be filled in with our machine learning algorithms.  Columns are automatically cleansed of poor entries, and NaN values in the input dataframe are automatically imputed for these machine learning algorithms to operate.  Categorical entries are automatically encoded and scaled for the machine learning algorithms.

The following machine learning models are utilized:  Random Forest, Support Vector Machines, K Nearest Neighbors, Gradient Boosting, Logistic/Linear Regression, and Adaboosting.  These models are weighted according to their performance on validation sets taken from the input data before being combined into an ensemble model.  All models are taken from sklearn in order to minimize the number of packages needed to run this code.

Below are the variables that can be specified when running the primary function (column_classification) in ClassificationAlgorithm.py:
df = input dataframe containing both filled and unfilled values in our target dataframe
column_name = column that we wish to fill in missing values for
cols_to_use = columns from input dataframe that we will use for determining classification, input as a list
  Default value is None, in which all columns are used (may lead to memory issues)
  A secondary option is to just input a list of columns
  A third option is to input a list of 3 lists; 1st for date columns, 2nd for categorical columns, 3rd for measure columns
    ex: [['DATE_COL1', 'DATE_COL2'], ['CAT_COL1'], ['MEASURE_COL1']]
cull_cols = option to perform column culling during fitting.  This will automatically determine which columns are helpful 
  for fitting our column of interest and exclude those that aren't, resulting in a better fit, but increases computation time
models = preselection of methods to use for classification/regression.  
  The potential inputs are 'RF', 'LSVC', 'KNN', 'GBC', 'LR', 'ADA'.  All methods are used by default.
  RF: Random Forest Model
  LSVC: Linear Support Vector Model
  KNN: K-Nearest Neighbors Model
  GBC: Gradient Boosting Model
  LR: Linear/Logistic Regression Model
  ADA: Adaboost Model
row_limit = limit on number of rows to use for training the models.  Prevents excessive runtimes on large datasets
class_thresh = classification threshold for using a model's prediction (values range from 0 to 1).
  Example: If random forest has a 95% confidence in its prediction, it has reached a class_thresh of 0.95
  If you set a class_thresh of 0.95, only models that are 95% confident (or higher) in their predictions will be used for classification
    Note:  This variable has no effect on regression models, and thus has no impact on filling date/number columns
    Note:  All input models having confidences > class_thresh will be utilized for filling values.  In other words, if you specify
      all 6 models are to be used, any row where at least one method has a confidence > class_thresh will be filled
