import pandas as pd
import numpy as np
import classificationUtils as cu
import datetime
import operator
import itertools
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFECV, RFE #RFECV slower, but can be more accurate
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error #Linear fit measure that gives less influence to outliers

#Options set to improve display
pd.options.display.float_format = '{:.2f}'.format #Suppressing scientific notation
pd.options.mode.chained_assignment = None #Disable some warnings from pandas
pd.set_option('display.max_columns', 250) #Set max number of columns to display in notebook

#Ensure directory exists for placing images
cu.check_directory('./figures')
cu.check_directory('./savedmodels')

#Track time to complete tasks throughout script
start_time = datetime.datetime.now()



#####################################################
#LOAD THE DATA
#data_folder = 'C:/SHARE/Data/'
data_folder = 'C:/Users/Administrator/Documents/EDD_UseCase/'
#data_filename = 'SDDB_Details_AFRICOM'
#data_filename = 'EUCOM_SDDB_DATA_OCTtoDEC2019_EmptyVals'
#data_filename = 'SDDB_Details_AFRICOM_EmptyVals'
#data_filename = 'I20200102_KAISER_PIVOT_RQST_DATA_EmptyVals'
data_filename = 'vSDDB_PIVOT_SOUTHCOM_DATA_Aggregated_MissingVal'
data_extension = '.csv'

#Read an excel file:
if data_extension == '.xlsx':
    try:
        df = pd.read_excel(data_folder + data_filename + data_extension)
        print('File loaded.')
    except:
        print('Error: Data file not found.')
    print("Load Data Run Time: ", datetime.datetime.now() - start_time)
#Read other file types (.csv, .txt)
else:
    try:
        df = cu.load_dataframe(data_folder, data_filename, data_extension)
        print('File loaded.')
    except:
        print('Error: Data file not found.')
    print("Load Data Run Time: ", datetime.datetime.now() - start_time)



#######################################################
#DEFINE THE CLASSIFICATION FUNCTION
def column_classification(df, column_name, cols_to_use = None, cull_cols = False, models = ['RF', 'LSVC', 'KNN', 'GBC', 'LR', 'ADA'], row_limit = 100000, class_thresh = 0):
    #This function aims to fill in the null/missing values in a column (thereby classifying those values)
    # This function does this by intaking a dataframe, looking at rows in the specified column with filled values, and
    # using the information in other columns to determine how to fill in the missing values
    #df = input dataframe containing both filled and unfilled values in our target dataframe
    #column_name = column that we wish to fill in missing values for
    #cols_to_use = columns from input dataframe that we will use for determining classification, input as a list
    # Default value is None, in which all columns are used (may lead to memory issues)
    # A secondary option is to just input a list of columns
    # A third option is to input a list of 3 lists; 1st for date columns, 2nd for categorical columns, 3rd for measure columns
    #   ex: [['DATE_COL1', 'DATE_COL2'], ['CAT_COL1'], ['MEASURE_COL1']]
    #cull_cols = option to perform column culling during fitting.  This will automatically determine which columns are helpful 
    # for fitting our column of interest and exclude those that aren't, resulting in a better fit, but increases computation time
    #models = preselection of methods to use for classification/regression.  
    # The potential inputs are 'RF', 'LSVC', 'KNN', 'GBC', 'LR', 'ADA'.  All methods are used by default.
    # RF: Random Forest Model
    # LSVC: Linear Support Vector Model
    # KNN: K-Nearest Neighbors Model
    # GBC: Gradient Boosting Model
    # LR: Linear/Logistic Regression Model
    # ADA: Adaboost Model
    #row_limit = limit on number of rows to use for training the models.  Prevents excessive runtimes on large datasets
    #class_thresh = classification threshold for using a model's prediction (values range from 0 to 1).
    # Example: If random forest has a 95% confidence in its prediction, it has reached a class_thresh of 0.95
    # If you set a class_thresh of 0.95, only models that are 95% confident (or higher) in their predictions will be used for classification
    # Note:  This variable has no effect on regression models, and thus has no impact on filling date/number columns
    # Note:  All input models having confidences > class_thresh will be utilized for filling values.  In other words, if you specify
    #  all 6 models are to be used, any row where at least one method has a confidence > class_thresh will be filled
    
    start_time = datetime.datetime.now()
    
    
    ##############################################################
    #FUNCTION SANITY CHECKS
    
    #Ensure cols_to_use input correctly
    dataframecolumnlist = list(df.columns.values)
    cols_to_use_type = None
    if cols_to_use != None:
        if type(cols_to_use) != list: #Ensure only a list is input
            print('Please input cols_to_use as None, a list of 3 lists, or a list of strings.')
            return None
        for entry in cols_to_use: #Ensure the input list is either a list of lists or strings
            if type(entry) != list and type(entry) != str: #Ensure cols_to_use is either a list of lists or list of strings
                print('Please input either a list of 3 lists or a list of strings for cols_to-use.')
                return None
        
        if type(cols_to_use[0]) == str: #If first entry is a string, all entries should be strings
            cols_to_use_type = 'str'
            if column_name in cols_to_use: #Ensure the column to classify isn't input here
                print('Do not input the column to analyze in the columns to use for analysis as well.')
                return None
            for entry in cols_to_use:
                if type(entry) != str:
                    print('If inputting a list of strings for cols_to_use, ensure all entries are strings.')
                    return None
                if entry not in dataframecolumnlist: #All entries should exist in the input dataframe
                    print('One of inputs in cols_to_use is not present in the dataframe: ' + str(entry))
                    
        if type(cols_to_use[0]) == list: #If first entry is a list, cols_to_use should be a list of 3 lists
            cols_to_use_type = 'list'
            if len(cols_to_use) != 3: #Ensure exactly 3 lists
                print('If inputting lists for cols_to_use, input exactly 3 lists for date, categorical, and measure columns, respectively.')
                return None
            for entry in cols_to_use:
                if type(entry) != list:
                    print('If inputting lists for cols_to_use, ensure all entries are lists.')
                    return None
                for item in entry:
                    if type(item) != str: #All entries in our 3 sublists should be strings
                        print('If inputting lists for cols_to_use, ensure all entries within the lists are strings')
                        return None
                    if item not in dataframecolumnlist: #All entries in our 3 sublists should exist in the input dataframe
                        print('One of the subinputs in cols_to_use is not present in the dataframe: ' + str(item))
                        return None
                    if item == column_name: #Ensure the column to classify isn't input here
                        print('Do not input the column to analyze in the columns to use for analysis as well.')
                        return None
                    
    #Ensure cull_cols entered correctly                
    if (cull_cols != True) and (cull_cols != False): #Ensure coll_cols is boolean
        print('Only input booleans (True or False) as values for cull_cols.')
        return None
    
    #Ensure models entered correctly
    if type(models) != list:
        print('Please input the methods variable as a list')
        return None
    for item in models:  #Ensure values in methods all reference available models 
        if item not in ['RF', 'LSVC', 'KNN', 'GBC', 'LR', 'ADA']:
            print("Option in methods unavailable. Only select among the following options for methods variable: 'RF', 'LSVC', 'KNN', 'GBC', 'LR', 'ABA'.")
            return None
    if len(models) < 0:
        print('Please input at least one method in the method variable')
        return None
    
    #Ensure row_limit entered correctly
    if type(row_limit) != int:
        print('Please only input integers for row_limit')
        return None
    
    #Ensure class_thresh entered correctly
    if type(class_thresh) != float and type(class_thresh) != int:
        print('Please only input values between 0 and 1 for class_thresh')
        return None
    if class_thresh < 0 or class_thresh > 1:
        print('Please only input values between 0 and 1 for class_thresh')
        return None
    
    #Reduce dataframe to only the columns specified
    if cols_to_use_type == 'str': #If cols_to_use input was a list of strings
        for column in df:
            if column not in cols_to_use and column != column_name:
                del df[column]
                MemUsage = df.memory_usage().sum() #Minimizes memory usage
    if cols_to_use_type == 'list': #If cols_to_use input was a list of 3 lists
        for column in df:
            checker = 0
            for entry in cols_to_use: #Loop over each of 3 lists
                for item in entry: #Loop over item in each list
                    if column == item:
                        checker = 1
            if column == column_name:
                checker = 1
            if checker == 0: #If df column was never one of the user input columns, remove it
                del df[column]
                MemUsage = df.memory_usage().sum() #Minimizes memory usage
                
    #Remove columns that are largely null (>25% null) before cleaning (too many nulls to help with fitting)
    if cols_to_use == None:
        for column in df:
            if column != column_name:
                if (pd.isnull(df[column]).sum()/len(df[column])) >= 0.25: #If over 25% of entries are null
                    MemUsage = df.memory_usage().sum() #Minimizes memory usage
                    del df[column]
                    print('Column removed due to excessive (>25%) NaN values: ' + str(column))    
        
            
    #Determine data types to allow for cleaning of dataframe
    #If data types were not submitted manually: For each column, determine data format, then add to appropriate list
    if cols_to_use == None or cols_to_use_type == 'str': #If no guidance on column types was given, automatically determine them
        df, measure_fields, date_fields, categorical_fields = cu.determine_dtypes(df)
        
        #Move column_name to end of its list (ensures the LabelEncoder works correctly)
        # Note: This is not really needed for date/measure fields to work correctly, but matches form if columns manually submitted
        if column_name in measure_fields:
            measure_fields.append(measure_fields.pop(measure_fields.index(column_name)))
        if column_name in date_fields:
            date_fields.append(date_fields.pop(date_fields.index(column_name)))
        if column_name in categorical_fields:
            categorical_fields.append(categorical_fields.pop(categorical_fields.index(column_name)))
                
    else: #If data column types were already determined/submitted, simply assign them to correct list
        date_fields = cols_to_use[0]
        categorical_fields = cols_to_use[1]
        measure_fields = cols_to_use[2]
        
        #Determine type for the column to analyze
        coldf = df[[column_name]]
        coldf, colmeas, coldate, colcat = cu.determine_dtypes(coldf)
        if colmeas == [column_name]: #If column to analyze is a measure
            measure_fields.append(column_name)
        if coldate == [column_name]:
            date_fields.append(column_name)
        if colcat == [column_name]:
            categorical_fields.append(column_name)
            
    #Ensure that if we have a class_thresh specified that we are working on classification of a categorical column
    if class_thresh > 0 and column_name in measure_fields:
        print('Column of interest, ' + str(column_name) + ' was found to be numerical.  However, a class_thresh > 0 was specified.')
        print('class_thresh can only be specified when filling categorical columns.  Please set class_thresh to 0 or select a new column.')
        return None
    if class_thresh > 0 and column_name in date_fields:
        print('Column of interest, ' + str(column_name) + ' was found to be datetime.  However, a class_thresh > 0 was specified.')
        print('class_thresh can only be specified when filling categorical columns.  Please set class_thresh to 0 or select a new column.')
        return None
        
    
    #Clean the dataframe
    #Date columns must always be cleaned to convert to datetime formatting
    print("Cleaning date data")
    for field in date_fields:
        cu.clean_date_data(df, [field])
        
    cleanfile = True #True by default, only switch if category/measure fields have already been cleaned
    if cleanfile == True: #If you want to clean the measure/category fields
        print("Cleaning measure data")
        for field in measure_fields:
            cu.clean_measure_data(df,[field])
        print("Cleaning categorical data")
        for field in categorical_fields:
            cu.clean_categorical_data(df, [field])
    print('Data cleaning completed.')
    print(datetime.datetime.now() - start_time)
    start_time = datetime.datetime.now()
    
    #If cleaning caused a column to become fully null, exclude it
    for column in df:
        #Check if column is entirely filled with NaN's.  If so, no point analyzing
        if pd.isnull(df[column]).sum() == len(df[column]):
            print('After cleaning, column is entirely NaN, excluding from analysis: ' + str(column))
            if column in measure_fields:
                measure_fields.remove(column)
            if column in categorical_fields:
                categorical_fields.remove(column)
            if column in date_fields:
                date_fields.remove(column)
            del df[column]
    
    #If column of interest is numeric with low number of unique values, make it categorical
    if column_name in measure_fields:
        if len(df[column_name].unique()) < 50: #If <50 unique values, column is likely better fitted as a category
            df[column_name] = df[column_name].astype('category')
            measure_fields.remove(column_name)
            categorical_fields.append(column_name)
            print('Treating column of interest, ' + str(column_name) + ' as a category due to low number of unique values')
    
    #Remove categorical fields with >100 unique entries (difficult for determining patterns)
    for column in categorical_fields:
        if column != column_name:
            if len(df[column].unique()) > 100:
                del df[column]
                categorical_fields.remove(column)
                print('Categorical column removed due to excessive (>100) unique values: ' + str(column))
                
    #Remove columns that are over 25% nulls (unlikely to help with fitting)
    if cols_to_use == None:
        for column in categorical_fields:
            if column != column_name:
                if (pd.isnull(df[column]).sum()/len(df[column])) >= 0.25: #If over 25% of entries are null
                    del df[column]
                    categorical_fields.remove(column)
                    print('Categorical column removed due to excessive (>25%) NaN values after data cleaning: ' + str(column))
        for column in measure_fields:
            if column != column_name:
                if (pd.isnull(df[column]).sum()/len(df[column])) >= 0.25: #If over 25% of entries are null
                    del df[column]
                    measure_fields.remove(column)
                    print('Measure column removed due to excessive (>25%) NaN values after data cleaning: ' + str(column))
        for column in date_fields:
            if column != column_name:
                if (pd.isnull(df[column]).sum()/len(df[column])) >= 0.25: #If over 25% of entries are null
                    del df[column]
                    date_fields.remove(column)
                    print('Date column removed due to excessive (>25%) NaN values after data cleaning: ' + str(column))
    
    #Option to save the file (in order to avoid cleaning/row deletion in future runs)
    savefile = False
    if savefile == True:
        cu.save_file(df, data_folder, data_filename, '_CLEAN.csv')
        print(datetime.datetime.now() - start_time)
        start_time = datetime.datetime.now()
        
    
    
    ####################################################
    #FURTHER DATA CLEANSING FOR USE IN MACHINE LEARNING MODELS
    
    #Fill numerical column NaNs with the mean for the column
    for column in measure_fields:
        if column != column_name:
            df[column] = df[column].fillna(df[column].mean())  
    #Fill categorical column NaNs with most popular option for the column
    for column in categorical_fields:
        print(column) ##
        if column != column_name:
            df[column].fillna(df[column].value_counts().idxmax(), inplace = True)
            #Alternate option:  Fill in with the string 'None'
            #df[column] = df[column].fillna('None')
    #Convert datetime to a number, fill NaNs with mean -> should be updated to match any other dates in the entry
    for column in date_fields:
        df[column] = (df[column] - datetime.datetime(1970,1,1)).dt.total_seconds() #Covert from date to seconds
        if column != column_name:
            #First, try to fill datetimes with mean of datetimes from the row (more likely to be accurate)
            df[column] = df[column].fillna(df[date_fields].mean(axis=1))
            #Second, fill remaining NaN's (due to the row having only null datetimes) with column means
            df[column] = df[column].fillna(df[column].mean())

    #Find indicies with NaN values to later separate test/train set
    train_index = df.index[pd.isnull(df[column_name]) == False]
    test_index = df.index[pd.isnull(df[column_name]) == True]
    test_index_list = test_index.tolist()
    print('Number of NaN values to classify in column ' + str(column_name) + ': ' + str(len(test_index)))
    
    #Encode categorical values 
    le = LabelEncoder()
    for column in categorical_fields: #Note: column_name will be last entry in this list, so saved encoder will work for unencoding y
        df[column] = le.fit_transform(df[column].astype(str)) 
    print('Coverted categorical columns into encoded integers')
    
    #Set limit on number of training rows to analyze (keeps training time down)
    #traindf = df.loc[train_index].head(row_limit) ##Only use first row_limit samples (rather than random ones)
    if row_limit > len(train_index): #If our limit is larger than our sample size, use all available samples
        traindf = df.loc[train_index] 
    if row_limit < len(train_index): #If our sample size is larger than our row limit, take a random sample
        traindf = df.loc[train_index].sample(n=row_limit)
    
    #If any categorical value only appears once, remove that row from training (cross val. can't be done w/ 1 value)
    traindf['freq'] = traindf.groupby(column_name)[column_name].transform('count')
    traindf = traindf.drop(traindf[traindf['freq'] == 1].index)
    del traindf['freq']
    
    #Define the test set
    y_train = traindf[[column_name]]
    if column_name in measure_fields or column_name in date_fields: #Scale y train if it's date or numeric
        yscaler = RobustScaler() 
        y_train[[column_name]] = yscaler.fit_transform(y_train[[column_name]]) 
    y_train = y_train[column_name].values 
    x_train = traindf.drop(column_name, axis = 1)
    x_train_cols = x_train.columns
    
    #Further split training set into training and validation set
    if cull_cols == False:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1)
        x_test = df.loc[test_index] #Select the indicies that had null values for column of interest
        x_test = x_test.drop(column_name, axis = 1)
        y_test = df.loc[test_index] 
        y_test = y_test[[column_name]]
    
    #Scale variables to be similar in scale
    scaler = RobustScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    if cull_cols == False:
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
    print('Columns scaled to similar sizes')
    print(datetime.datetime.now() - start_time)
    start_time = datetime.datetime.now() 
    
    
    #Check if y_train only has a single class; if only a single class present, there's no point in using machine learning techniques
    if column_name in categorical_fields:
        y_class_count = len(np.unique(y_train))
        print('Number of different classes in training set:  ' + str(y_class_count))
        if y_class_count == 1:
            print('Only one class is present in the training set.  Classification techniques will not be useful; all predictions will be the one '
                  'represented class.  Either improve the training set, or simply fill the missing values with the value of the single class.')
            print('Returning the original dataframe with the column now filled with the one class.')
            
            #Reload the original dataframe (before we filled NaNs and mangled it in other ways)
            df = cu.load_dataframe(data_folder, data_filename, data_extension)
            #Fill null values with the single present class in the training set
            if column_name in date_fields:
                cu.clean_date_data(df, [column_name])
            if column_name in measure_fields:
                cu.clean_measure_data(df, [column_name])
            if column_name in categorical_fields:
                cu.clean_categorical_data(df, [column_name])
            ndf = df.loc[df[column_name].notnull()]
            only_value = np.unique(ndf[column_name])[0]
            df.loc[test_index,column_name] = only_value
            
            #Save the dataframe now filled with our predictions for the column
            print('Saving dataframe with predictions entered, denoted FILLED')
            cu.save_file(df, data_folder, data_filename, '_FILLED.csv')
            return None
        

    
    #########################################################
    #DETERMINATION OF USEFUL COLUMNS
    #Here we use a random forest model to determine which columns are actually helpful (at least 
    # show some correlation) with our desired column to fill.  Unhelpful columns are removed 
    # from the training and test sets.  This is done using recursive feature elimination (RFE).
    
    if cull_cols == True:
        if column_name in categorical_fields: #If column is categorical
            print('Using Random Forest Classification Model to cull unhelpful columns')
            rf = RandomForestClassifier(n_estimators=100)
        else: #If column is numeric or datetime
            print('Using Random Forest Regression Model to cull unhelpful columns')
            rf = RandomForestRegressor(n_estimators=100)
        #feature_eliminator = RFECV(rf, verbose=1, n_jobs = -1, cv=2) #Slower, but potentially more accurate
        feature_eliminator = RFE(rf, verbose=1) 
        feature_eliminator.fit(x_train, y_train)
        useful_cols = [f for f,s in zip(x_train_cols, feature_eliminator.support_) if s] #Turn useful_cols into list
        print('Useful columns: ', useful_cols)
        print('Number of useful columns: ', len(useful_cols))
        
        #Create new train and test sets using the useful columns
        x_train = traindf[useful_cols]
        x_test = df.loc[test_index]
        x_test = x_test[useful_cols]
        y_test = df.loc[test_index] #Select the indicies that had null values for column of interest
        y_test = y_test[[column_name]]       
        
        #Further split training set into training and validation set 
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1) 
        
        #Scale variables to be similar in scale
        scaler = RobustScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val) 
        
        print('Unhelpful columns eliminated, new train and test set generated')
        print(datetime.datetime.now() - start_time)
        start_time = datetime.datetime.now() 
        
    #Unencode y_val for later use in results dataframe
    if column_name in categorical_fields:
        y_val_unenc = le.inverse_transform(y_val) #For testing performance of our algorithms on val set
    
    #Initialize dataframes to add our prediction results to
    allpreds = pd.DataFrame({'Init': np.nan}, index=test_index) 
    if column_name in categorical_fields:
        valpreds = pd.DataFrame({'Actual': y_val_unenc}) 
    else:
        valpreds = pd.DataFrame({'Actual': y_val})
    ModelScores = pd.DataFrame({'Init': np.nan}, index=[0]) 
    #Initialize variable trackers for best model/ensemble
    bestscore = 0
    bestcorr = 0
    bestmodel = []
    
    
    
    ############################################################
    #RANDOM FOREST MODEL IMPLEMENTATION
    
    if 'RF' in models:
        print('Beginning Random Forest Model')
        if column_name in categorical_fields:  #If column is categorical
            rf = RandomForestClassifier(n_estimators=150, oob_score=True, max_depth = len(x_train[0])//1.5) #Define model, limit depth to prevent overfits
            rf.fit(x_train, y_train) #Fit data to random forest model
            
            rfvalpred = rf.predict(x_val) #Use random forest to make predictions on our validation set
            rfvalpredprob = rf.predict_proba(x_val) #Find the confidences in each prediction
            rfpred = rf.predict(x_test) 
            rfpredprob = rf.predict_proba(x_test) 
            
            rfvalpred = le.inverse_transform(rfvalpred) #Unencode our predictions
            rfpred = le.inverse_transform(rfpred) 
            
            #Save our predictions (validation set and test set) to separate dataframes
            y_val_rf = pd.DataFrame({'RF_Pred': rfvalpred}) 
            y_test_rf = pd.DataFrame({'RF_Pred': rfpred}, index=test_index) 
            y_val_rf['RF_Conf'] = np.nan
            y_test_rf['RF_Conf'] = np.nan
            rfvalpredprob = rfvalpredprob.tolist()
            rfpredprob = rfpredprob.tolist()
            for i in range(len(rfpredprob)):
                y_test_rf['RF_Conf'][test_index_list[i]] = max(rfpredprob[i])
            for i in range(len(rfvalpredprob)):
                y_val_rf['RF_Conf'][i] = max(rfvalpredprob[i])
            
            #Save the fit model
            joblib.dump(rf, './savedmodels/RF_Classifier.pkl') #Note: Reload the model with joblib.load()
            
        else:  #If column is numeric or datetime
            rf = RandomForestRegressor(n_estimators=150, oob_score=True, max_depth = len(x_train[0])//1.5)
            rf.fit(x_train, y_train)
            
            rankimportances = True
            if rankimportances == True: #Rank columns by importance to correct classification
                rfimportances = rf.feature_importances_
                rfstd = np.std([tree.feature_importances_ for tree in rf.estimators_], axis = 0)

                rfi, x_tr = zip(*sorted(zip(rfimportances, x_train_cols), reverse = True))
                print("Feature ranking:")
                for f in range(x_train.shape[1]):
                    print(str(f) + ".  Feature " + str(x_tr[f]) + "  (" + str(round(rfi[f], 4)) + ")")
            
            rfvalpred = rf.predict(x_val)
            rfpred = rf.predict(x_test)
            
            y_val_rf = pd.DataFrame({'RF_Pred': rfvalpred})
            y_test_rf = pd.DataFrame({'RF_Pred': rfpred}, index=test_index) 
            
            #Save the fit model
            joblib.dump(rf, './savedmodels/RF_Regressor.pkl')
            
        rf_oob_score = rf.oob_score_
        rf_score = round(rf.score(x_train, y_train),4)
        rf_val_score = round(rf.score(x_val, y_val),4) 
        
        print('Random Forest Model training completed')
        print('Random Forest Model Oob (Out-of-bag) score: ', round(rf_oob_score,4))
        print('Random Forest Model score: ', rf_score)
        print('Random Forest Model val score: ', rf_val_score) 
        
        #Add results to our dataframes storing prediction/scoring results    
        allpreds = pd.concat([allpreds, y_test_rf], axis=1) 
        valpreds = pd.concat([valpreds, y_val_rf], axis=1)
        rf_mscores = pd.DataFrame({'RF_Score': rf_val_score}, index=[0])
        ModelScores = pd.concat([ModelScores, rf_mscores], axis = 1)
        
        if class_thresh > 0: #Print results accounting for our class_thresh value
            rf_vals_above = valpreds['RF_Conf'][valpreds['RF_Conf'] > class_thresh].count()
            rf_vals_above_frac = rf_vals_above/valpreds.shape[0]
            print('On validation set, fraction of values filled (RF confidence > ' + str(class_thresh) + '):  ' + str(round(rf_vals_above_frac,4)))
            rf_filled_match = valpreds['RF_Pred'][(valpreds['RF_Conf'] > class_thresh) & (valpreds['Actual'] == valpreds['RF_Pred'])].count()
            rf_filled_acc = rf_filled_match/rf_vals_above
            print('Accuracy of RF on predictions with confidences above ' + str(class_thresh) + ':  ' + str(round(rf_filled_acc,4)))
        
        #Track best performing model
        if rf_val_score > bestscore: 
            bestscore = rf_val_score
            bestmodel = ['RF']
            bestvals = valpreds[['RF_Pred']]
            if class_thresh > 0:
                bestnumfilled = rf_vals_above
                bestvalmatches = rf_filled_match
                
        print(datetime.datetime.now() - start_time)
        start_time = datetime.datetime.now()  
    
        
    #########################################
    #SUPPORT VECTOR MACHINES
    
    if 'LSVC' in models:
        print('Beginning Support Vector Machine Model')
        if column_name in categorical_fields:  #If column is categorical
            svc = SVC(gamma = 'scale', max_iter=-1)
            lsvc = CalibratedClassifierCV(svc, cv = 2) #This step is needed for predict_proba function to be accessible
            lsvc.fit(x_train, y_train)
            
            lsvcvalpred = lsvc.predict(x_val) #Use lsvc to make predictions on our validation set
            lsvcvalpredprob = lsvc.predict_proba(x_val) #Find the confidences in each prediction
            lsvcpred = lsvc.predict(x_test)
            lsvcpredprob = lsvc.predict_proba(x_test)
            
            lsvcvalpred = le.inverse_transform(lsvcvalpred) #Unencode our predictions
            lsvcpred = le.inverse_transform(lsvcpred)
            
            #Save our predictions (validation set and test set) to separate dataframes
            y_val_lsvc = pd.DataFrame({'LSVC_Pred': lsvcvalpred})
            y_test_lsvc = pd.DataFrame({'LSVC_Pred': lsvcpred}, index=test_index) 
            y_val_lsvc['LSVC_Conf'] = np.nan
            y_test_lsvc['LSVC_Conf'] = np.nan
            lsvcvalpredprob = lsvcvalpredprob.tolist()
            lsvcpredprob = lsvcpredprob.tolist()
            
            for i in range(len(lsvcpredprob)):
                y_test_lsvc['LSVC_Conf'][test_index_list[i]] = max(lsvcpredprob[i])
            for i in range(len(lsvcvalpredprob)):
                y_val_lsvc['LSVC_Conf'][i] = max(lsvcvalpredprob[i])
                
            #Save the fit model
            joblib.dump(lsvc, './savedmodels/LSVC_Classifier.pkl')
            
        else: #If column is numeric or datetime
            lsvc = SVR(gamma = 'scale', max_iter=-1)
            lsvc.fit(x_train, y_train)
            
            lsvcvalpred = lsvc.predict(x_val)
            lsvcpred = lsvc.predict(x_test)
            
            y_val_lsvc = pd.DataFrame({'LSVC_Pred': lsvcvalpred})
            y_test_lsvc = pd.DataFrame({'LSVC_Pred': lsvcpred}, index=test_index) 
            
            #Save the fit model
            joblib.dump(lsvc, './savedmodels/LSVC_Regressor.pkl')
            
        lsvc_score = round(lsvc.score(x_train, y_train),4)
        lsvc_val_score = round(lsvc.score(x_val, y_val),4)
        
        print('Support Vector Model training completed')
        print('Support Vector Model score: ', lsvc_score)
        print('Support Vector Model val score: ', lsvc_val_score) 
        
        #Add results to our dataframes storing prediction/scoring results    
        allpreds = pd.concat([allpreds, y_test_lsvc], axis=1) 
        valpreds = pd.concat([valpreds, y_val_lsvc], axis=1)
        lsvc_mscores = pd.DataFrame({'LSVC_Score': lsvc_val_score}, index=[0])
        ModelScores = pd.concat([ModelScores, lsvc_mscores], axis = 1)
        
        if class_thresh > 0: #Print results accounting for our class_thresh value
            lsvc_vals_above = valpreds['LSVC_Conf'][valpreds['LSVC_Conf'] > class_thresh].count()
            lsvc_vals_above_frac = lsvc_vals_above/valpreds.shape[0]
            print('On validation set, fraction of values filled (LSVC confidence > ' + str(class_thresh) + '):  ' + str(round(lsvc_vals_above_frac,4)))
            lsvc_filled_match = valpreds['LSVC_Pred'][(valpreds['LSVC_Conf'] > class_thresh) & (valpreds['Actual'] == valpreds['LSVC_Pred'])].count()
            lsvc_filled_acc = lsvc_filled_match/lsvc_vals_above
            print('Accuracy of LSVC on predictions with confidences above ' + str(class_thresh) + ':  ' + str(round(lsvc_filled_acc,4)))
            
        #Track best performing model
        if lsvc_val_score > bestscore: 
            bestscore = lsvc_val_score
            bestmodel = ['LSVC']
            bestvals = valpreds[['LSVC_Pred']]
            if class_thresh > 0:
                bestnumfilled = lsvc_vals_above
                bestvalmatches = lsvc_filled_match            
        
        print(datetime.datetime.now() - start_time)
        start_time = datetime.datetime.now()
        
    
    ##########################################
    #K NEAREST NEIGHBORS CLASSIFIER
    
    if 'KNN' in models:
        print('Beginning K-Nearest Neighbors Model')
        if column_name in categorical_fields:  #If column is categorical
            knn = KNeighborsClassifier(n_neighbors = 5, n_jobs = -1)
            knn.fit(x_train, y_train)
            
            knnvalpred = knn.predict(x_val)
            knnvalpredprob = knn.predict_proba(x_val)
            knnpred = knn.predict(x_test)
            knnpredprob = knn.predict_proba(x_test)
            
            knnvalpred = le.inverse_transform(knnvalpred) #Unencode our predictions
            knnpred = le.inverse_transform(knnpred)
            
            #Save our predictions (validation set and test set) to separate dataframes
            y_val_knn = pd.DataFrame({'KNN_Pred': knnvalpred})
            y_test_knn = pd.DataFrame({'KNN_Pred': knnpred}, index=test_index) 
            y_val_knn['KNN_Conf'] = np.nan
            y_test_knn['KNN_Conf'] = np.nan
            knnpredprob = knnpredprob.tolist()
            knnvalpredprob = knnvalpredprob.tolist()
            for i in range(len(knnpredprob)):
                y_test_knn['KNN_Conf'][test_index_list[i]] = max(knnpredprob[i])
            for i in range(len(knnvalpredprob)):
                y_val_knn['KNN_Conf'][i] = max(knnvalpredprob[i])
                
            #Save the fit model
            joblib.dump(knn, './savedmodels/KNN_Classifier.pkl')
            
        else: #If column is numeric or datetime
            knn = KNeighborsRegressor(n_neighbors = 10, n_jobs = -1)
            knn.fit(x_train, y_train)
            
            knnvalpred = knn.predict(x_val)
            knnpred = knn.predict(x_test) 
            
            y_val_knn = pd.DataFrame({'KNN_Pred': knnvalpred})
            y_test_knn = pd.DataFrame({'KNN_Pred': knnpred}, index=test_index)
            
            #Save the fit model
            joblib.dump(knn, './savedmodels/KNN_Regressor.pkl')
            
        knn_score = round(knn.score(x_train, y_train),4)
        knn_val_score = round(knn.score(x_val, y_val),4)
    
        print('K-Nearest Neighbors Model training completed')
        print('K-Nearest Neighbors Model score: ', knn_score)
        print('K-Nearest Neighbors Model val score: ', knn_val_score)
        
        #Add results to our dataframe storing prediction results    
        allpreds = pd.concat([allpreds, y_test_knn], axis=1) 
        valpreds = pd.concat([valpreds, y_val_knn], axis=1)
        knn_mscores = pd.DataFrame({'KNN_Score': knn_val_score}, index=[0])
        ModelScores = pd.concat([ModelScores, knn_mscores], axis = 1)
        
        if class_thresh > 0: #Print results accounting for our class_thresh value
            knn_vals_above = valpreds['KNN_Conf'][valpreds['KNN_Conf'] > class_thresh].count()
            knn_vals_above_frac = knn_vals_above/valpreds.shape[0]
            print('On validation set, fraction of values filled (KNN confidence > ' + str(class_thresh) + '):  ' + str(round(knn_vals_above_frac,4)))
            knn_filled_match = valpreds['KNN_Pred'][(valpreds['KNN_Conf'] > class_thresh) & (valpreds['Actual'] == valpreds['KNN_Pred'])].count()
            knn_filled_acc = knn_filled_match/knn_vals_above
            print('Accuracy of KNN on predictions with confidences above ' + str(class_thresh) + ':  ' + str(round(knn_filled_acc,4)))
            
        #Track best performing model
        if knn_val_score > bestscore: 
            bestscore = knn_val_score
            bestmodel = ['KNN']
            bestvals = valpreds[['KNN_Pred']]
            if class_thresh > 0:
                bestnumfilled = knn_vals_above
                bestvalmatches = knn_filled_match 
        
        print(datetime.datetime.now() - start_time)
        start_time = datetime.datetime.now()
    
    
    ############################################
    #GRADIENT BOOSTING CLASSIFIER
    
    if 'GBC' in models:
        print('Beginning Gradient Boosting Model')
        if column_name in categorical_fields:  #If column is categorical
            gbc = GradientBoostingClassifier(n_estimators = 100, max_depth = 4)
            gbc.fit(x_train, y_train)
            
            gbcvalpred = gbc.predict(x_val)
            gbcvalpredprob = gbc.predict_proba(x_val)
            gbcpred = gbc.predict(x_test)
            gbcpredprob = gbc.predict_proba(x_test)
            
            gbcvalpred = le.inverse_transform(gbcvalpred) #Unencode our predictions
            gbcpred = le.inverse_transform(gbcpred)
            
            #Save our predictions (validation set and test set) to separate dataframes
            y_val_gbc = pd.DataFrame({'GBC_Pred': gbcvalpred})
            y_test_gbc = pd.DataFrame({'GBC_Pred': gbcpred}, index=test_index) 
            y_val_gbc['GBC_Conf'] = np.nan
            y_test_gbc['GBC_Conf'] = np.nan
            gbcvalpredprob = gbcvalpredprob.tolist()
            gbcpredprob = gbcpredprob.tolist()
            for i in range(len(gbcpredprob)):
                y_test_gbc['GBC_Conf'][test_index_list[i]] = max(gbcpredprob[i])
            for i in range(len(gbcvalpredprob)):
                y_val_gbc['GBC_Conf'][i] = max(gbcvalpredprob[i])
                
            #Save the fit model
            joblib.dump(gbc, './savedmodels/GBC_Classifier.pkl')
            
        else: #If column is numeric or datetime
            gbc = GradientBoostingRegressor(n_estimators = 100, max_depth = 4)
            gbc.fit(x_train, y_train)
            
            gbcvalpred = gbc.predict(x_val)
            gbcpred = gbc.predict(x_test)
            
            y_val_gbc = pd.DataFrame({'GBC_Pred': gbcvalpred})
            y_test_gbc = pd.DataFrame({'GBC_Pred': gbcpred}, index=test_index)
            
            #Save the fit model
            joblib.dump(gbc, './savedmodels/GBC_Regressor.pkl')
            
        gbc_score = round(gbc.score(x_train, y_train),4)
        gbc_val_score = round(gbc.score(x_val, y_val),4) 
        
        print('Gradient Boosting Model training completed')
        print('Gradient Boosting Model score: ', gbc_score)
        print('Gradient Boosting Model val score: ', gbc_val_score) 
        
        #Add results to our dataframe storing prediction results    
        allpreds = pd.concat([allpreds, y_test_gbc], axis=1) 
        valpreds = pd.concat([valpreds, y_val_gbc], axis=1)
        gbc_mscores = pd.DataFrame({'GBC_Score': gbc_val_score}, index=[0])
        ModelScores = pd.concat([ModelScores, gbc_mscores], axis = 1)
        
        if class_thresh > 0: #Print results accounting for our class_thresh value
            gbc_vals_above = valpreds['GBC_Conf'][valpreds['GBC_Conf'] > class_thresh].count()
            gbc_vals_above_frac = gbc_vals_above/valpreds.shape[0]
            print('On validation set, fraction of values filled (GBC confidence > ' + str(class_thresh) + '):  ' + str(round(gbc_vals_above_frac,4)))
            gbc_filled_match = valpreds['GBC_Pred'][(valpreds['GBC_Conf'] > class_thresh) & (valpreds['Actual'] == valpreds['GBC_Pred'])].count()
            gbc_filled_acc = gbc_filled_match/gbc_vals_above
            print('Accuracy of GBC on predictions with confidences above ' + str(class_thresh) + ':  ' + str(round(gbc_filled_acc,4)))
            
        #Track best performing model
        if gbc_val_score > bestscore: 
            bestscore = gbc_val_score
            bestmodel = ['GBC']
            bestvals = valpreds[['GBC_Pred']]
            if class_thresh > 0:
                bestnumfilled = gbc_vals_above
                bestvalmatches = gbc_filled_match  
        
        print(datetime.datetime.now() - start_time)
        start_time = datetime.datetime.now()
    
    
    ###############################################
    #LOGISTIC/LINEAR REGRESSION CLASSIFIER
    
    if 'LR' in models:
        print('Beginning Logistic/Linear Regression Model')
        if column_name in categorical_fields:  #If column is categorical
            lr = LogisticRegression(solver = 'lbfgs', multi_class = 'auto', max_iter = 500, C = 1.0)
            lr.fit(x_train, y_train)
            
            lrvalpred = lr.predict(x_val)
            lrvalpredprob = lr.predict_proba(x_val)
            lrpred = lr.predict(x_test)
            lrpredprob = lr.predict_proba(x_test)
            
            lrvalpred = le.inverse_transform(lrvalpred) #Unencode our predictions
            lrpred = le.inverse_transform(lrpred)
            
            #Save our predictions (validation set and test set) to separate dataframes
            y_val_lr = pd.DataFrame({'LR_Pred': lrvalpred})
            y_test_lr = pd.DataFrame({'LR_Pred': lrpred}, index=test_index) 
            y_val_lr['LR_Conf'] = np.nan
            y_test_lr['LR_Conf'] = np.nan
            lrvalpredprob = lrvalpredprob.tolist()
            lrpredprob = lrpredprob.tolist()
            for i in range(len(lrpredprob)):
                y_test_lr['LR_Conf'][test_index_list[i]] = max(lrpredprob[i])
            for i in range(len(lrvalpredprob)):
                y_val_lr['LR_Conf'][i] = max(lrvalpredprob[i])
                
            #Save the fit model
            joblib.dump(lr, './savedmodels/LR_Classifier.pkl')
            
        else: #If column is numeric or datetime
            lr = LinearRegression()
            lr.fit(x_train, y_train)
            
            lrvalpred = lr.predict(x_val)
            lrpred = lr.predict(x_test)
            
            y_val_lr = pd.DataFrame({'LR_Pred': lrvalpred})
            y_test_lr = pd.DataFrame({'LR_Pred': lrpred}, index=test_index) 
            
            #Save the fit model
            joblib.dump(lr, './savedmodels/LR_Regressor.pkl')
            
        lr_score = round(lr.score(x_train, y_train),4)
        lr_val_score = round(lr.score(x_val, y_val),4) 
        
        print('Logistic/Linear Regression Model training completed')
        print('Logistic/Linear Regression Model score: ', lr_score)
        print('Logistic/Linear Regression Model val score: ', lr_val_score)
        
        #Add results to our dataframe storing prediction results    
        allpreds = pd.concat([allpreds, y_test_lr], axis=1) 
        valpreds = pd.concat([valpreds, y_val_lr], axis=1)
        lr_mscores = pd.DataFrame({'LR_Score': lr_val_score}, index=[0])
        ModelScores = pd.concat([ModelScores, lr_mscores], axis = 1)
        
        if class_thresh > 0: #Print results accounting for our class_thresh value
            lr_vals_above = valpreds['LR_Conf'][valpreds['LR_Conf'] > class_thresh].count()
            lr_vals_above_frac = lr_vals_above/valpreds.shape[0]
            print('On validation set, fraction of values filled (LR confidence > ' + str(class_thresh) + '):  ' + str(round(lr_vals_above_frac,4)))
            lr_filled_match = valpreds['LR_Pred'][(valpreds['LR_Conf'] > class_thresh) & (valpreds['Actual'] == valpreds['LR_Pred'])].count()
            lr_filled_acc = lr_filled_match/lr_vals_above
            print('Accuracy of LR on predictions with confidences above ' + str(class_thresh) + ':  ' + str(round(lr_filled_acc,4)))
            
        #Track best performing model
        if lr_val_score > bestscore: 
            bestscore = lr_val_score
            bestmodel = ['LR']
            bestvals = valpreds[['LR_Pred']]
            if class_thresh > 0:
                bestnumfilled = lr_vals_above
                bestvalmatches = lr_filled_match
        
        print(datetime.datetime.now() - start_time)
        start_time = datetime.datetime.now()
    
    
    ###################################################
    #ADABOOST CLASSIFIER
    
    if 'ADA' in models:
        print('Beginning Adaboost Model')
        if column_name in categorical_fields:  #If column is categorical
            ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), algorithm="SAMME.R", n_estimators=150)
            ada.fit(x_train, y_train)
            
            adavalpred = ada.predict(x_val)
            adapred = ada.predict(x_test)
            adavalpredprob = ada.predict_proba(x_val)
            adapredprob = ada.predict_proba(x_test)
            
            adavalpred = le.inverse_transform(adavalpred) #Unencode our predictions
            adapred = le.inverse_transform(adapred) 
            
            #Save our predictions (validation set and test set) to separate dataframes
            y_val_ada = pd.DataFrame({'ADA_Pred': adavalpred})
            y_test_ada = pd.DataFrame({'ADA_Pred': adapred}, index=test_index)
            y_val_ada['ADA_Conf'] = np.nan
            y_test_ada['ADA_Conf'] = np.nan
            adavalpredprob = adavalpredprob.tolist()
            adapredprob = adapredprob.tolist()
            for i in range(len(adapredprob)):
                y_test_ada['ADA_Conf'][test_index_list[i]] = max(adapredprob[i])
            for i in range(len(adavalpredprob)):
                y_val_ada['ADA_Conf'][i] = max(adavalpredprob[i])
                
            #Save the fit model
            joblib.dump(ada, './savedmodels/ADA_Classifier.pkl')
                
        else:  #If column is numeric or datetime
            ada = AdaBoostRegressor(n_estimators = 150)
            ada.fit(x_train, y_train)
            
            adavalpred = ada.predict(x_val)
            adapred = ada.predict(x_test)
            
            y_val_ada = pd.DataFrame({'ADA_Pred': adavalpred})
            y_test_ada = pd.DataFrame({'ADA_Pred': adapred}, index=test_index)
            
            #Save the fit model
            joblib.dump(ada, './savedmodels/ADA_Regressor.pkl')
            
        ada_score = round(ada.score(x_train, y_train),4)
        ada_val_score = round(ada.score(x_val, y_val),4) 

        print('Adaboost Model training completed')
        print('Adaboost Model score: ', ada_score)
        print('Adaboost Model val score: ', ada_val_score) 
        
        #Add results to our dataframe storing prediction results    
        allpreds = pd.concat([allpreds, y_test_ada], axis=1) 
        valpreds = pd.concat([valpreds, y_val_ada], axis=1)
        ada_mscores = pd.DataFrame({'ADA_Score': ada_val_score}, index=[0])
        ModelScores = pd.concat([ModelScores, ada_mscores], axis = 1)
        
        if class_thresh > 0: #Print results accounting for our class_thresh value
            ada_vals_above = valpreds['ADA_Conf'][valpreds['ADA_Conf'] > class_thresh].count()
            ada_vals_above_frac = ada_vals_above/valpreds.shape[0]
            print('On validation set, fraction of values filled (ADA confidence > ' + str(class_thresh) + '):  ' + str(round(ada_vals_above_frac,4)))
            ada_filled_match = valpreds['ADA_Pred'][(valpreds['ADA_Conf'] > class_thresh) & (valpreds['Actual'] == valpreds['ADA_Pred'])].count()
            ada_filled_acc = ada_filled_match/ada_vals_above
            print('Accuracy of ADA on predictions with confidences above ' + str(class_thresh) + ':  ' + str(round(ada_filled_acc,4)))
            
        #Track best performing model
        if ada_val_score > bestscore: 
            bestscore = ada_val_score
            bestmodel = ['ADA']
            bestvals = valpreds[['ADA_Pred']]
            if class_thresh > 0:
                bestnumfilled = ada_vals_above
                bestvalmatches = ada_filled_match    
        
        print(datetime.datetime.now() - start_time)
        start_time = datetime.datetime.now()
    
    
    ###################################################
    #FINAL ADJUSTMENTS TO PREDICTION/RESULTS DATAFRAMES
    del allpreds['Init']
    del ModelScores['Init']
    allpreds = allpreds.round(4) #Round values to 4 decimal places
    
    print('Saving model scores, denoted MSCORES')
    cu.save_file(ModelScores, data_folder, data_filename, '_MSCORES.csv')
    
    
    ###################################################
    #PLOT CONFIDENCE DISTRIBUTIONS FOR EACH MODEL
    
    NumPlots = len(list(allpreds.columns.values))//2 #Number of plots is half the number of allpreds cols
    
    #Confidence distributions are only present if column of interest is categorical
    if column_name in categorical_fields:  #If column is categorical
        fig = plt.figure(figsize=(20,NumPlots*3.5))
        plt.subplots_adjust(wspace=0.1, hspace=0.75)
        
        for i in range(0,NumPlots):
            ax=fig.add_subplot(NumPlots,1,i+1) #Of NumPlots total plots, create i'th plot
            ax.set_title(list(allpreds.columns.values)[(i*2)+1], fontsize=24)
            plt.setp(ax.get_xticklabels(), ha="right", rotation=35, fontsize=20)
            plt.setp(ax.get_yticklabels(), ha="right", fontsize=20)
            ax.hist(allpreds[list(allpreds.columns.values)[(i*2)+1]], 
                    bins=[0,0.05,0.10,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
        
        fig.savefig("./figures/ConfidencePlots.png", dpi = 50)
        plt.close('all')
    
    
    ###################################################
    #DETERMINE BEST MODEL OR BEST COMBINATION OF MODELS; USE THESE FOR PREDICTIONS
    
    #Ensure all of our previously determined scores make sense for reaching our final predictions
    if 'RF' in models:
        if rf_val_score < 0 or rf_val_score > 1:
            print('Random Forest Model val score was not between 0 and 1, set to 0 for predictions')
            rf_val_score = 0 #If score is 0, this model will not influence our predictions
    if 'LSVC' in models:
        if lsvc_val_score < 0 or lsvc_val_score > 1:
            print('Linear Support Vector Model val score was not between 0 and 1, set to 0 for predictions')
            lsvc_val_score = 0
    if 'KNN' in models:
        if knn_val_score < 0 or knn_val_score > 1:
            print('K-Nearest Neighbors Model val score was not between 0 and 1, set to 0 for predictions')
            knn_val_score = 0
    if 'GBC' in models:
        if gbc_val_score < 0 or gbc_val_score > 1:
            print('Gradient Boosting Model val score was not between 0 and 1, set to 0 for predictions')
            gbc_val_score = 0
    if 'LR' in models:
        if lr_val_score < 0 or lr_val_score > 1:
            print('Logistic/Linear Regression Model val score was not between 0 and 1, set to 0 for predictions')
            lr_val_score = 0
    if 'ADA' in models:
        if ada_val_score < 0 or ada_val_score > 1:
            print('Adaboost Model val score was not between 0 and 1, set to 0 for predictions')
            ada_val_score = 0

    #Determine which method/values to use to fill the nulls
    if column_name in categorical_fields:  #If column is categorical
        #Method:  For each null value, the 5 methods each have a prediction and confidence score (0-1)
        # Additionally, each method has a score representing its overall performance (0-1)
        # The weight assigned to each model's answer is it's confidence score * overall performance
        # Whichever answer has the highest score will be used to fill in the null value.
        value_list = [] #Initialize value tracker
        vallength = valpreds.shape[0] #Number of rows in valpreds
        
        #Print best individual model score so far to serve as a baseline for ensemble combinations
        print('Best individual model: ' + str(bestmodel[0]))
        print('Best individual model score: ' + str(bestscore))
        
        if len(models) == 1: #If only running with a single model
            if class_thresh == 0:
                if 'RF' in models:
                    allpreds['Final_Pred'] = allpreds['RF_Pred']
                if 'LSVC' in models:
                    allpreds['Final_Pred'] = allpreds['LSVC_Pred']
                if 'KNN' in models:
                    allpreds['Final_Pred'] = allpreds['KNN_Pred']
                if 'GBC' in models:
                    allpreds['Final_Pred'] = allpreds['GBC_Pred']
                if 'LR' in models:
                    allpreds['Final_Pred'] = allpreds['LR_Pred']
                if 'ADA' in models:
                    allpreds['Final_Pred'] = allpreds['ADA_Pred']
            else: #If a threshold value has been entered for classification predictions
                allpreds['Final_Pred'] = np.nan
                for index, row in allpreds.iterrows():
                    if 'RF' in models:
                        if allpreds['RF_Conf'][index] > class_thresh: #If model is sufficiently confident, enter our prediction
                            allpreds['Final_Pred'][index] = allpreds['RF_Pred'][index]
                    if 'LSVC' in models:
                        if allpreds['LSVC_Conf'][index] > class_thresh:
                            allpreds['Final_Pred'][index] = allpreds['LSVC_Pred'][index]
                    if 'KNN' in models:
                        if allpreds['KNN_Conf'][index] > class_thresh:
                            allpreds['Final_Pred'][index] = allpreds['KNN_Pred'][index]
                    if 'GBC' in models:
                        if allpreds['GBC_Conf'][index] > class_thresh:
                            allpreds['Final_Pred'][index] = allpreds['GBC_Pred'][index]
                    if 'LR' in models:
                        if allpreds['LR_Conf'][index] > class_thresh:
                            allpreds['Final_Pred'][index] = allpreds['LR_Pred'][index]
                    if 'ADA' in models:
                        if allpreds['ADA_Conf'][index] > class_thresh:
                            allpreds['ADA_Pred'][index] = allpreds['ADA_Pred'][index]
                       
            if class_thresh > 0:
                print('Number of values filled with confidences above ' + str(class_thresh) + ':  ' + str(allpreds['Final_Pred'].count()))
                print('Number of values not filled (confidences too low):  ' + str(allpreds['Final_Pred'].isna().sum()))
        
        #Iterate over all combinations of 2 or more models
        if len(models) > 1: #If more than one model was selected with no threshold
            for i in range(2, len(models) + 1): #Iterate over all combinations of submitted models
                modelslist = list(itertools.combinations(models,i))
                for entry in modelslist:
                    print('Testing combination of models: ' + str(entry))
                    
                    value_list = [] #Initialize value tracker
                    if class_thresh > 0:
                        val_filled = 0 #Initialize tracker of whether value was filled despite class_thresh threshold
                        val_match = 0 #Initialize tracker of whether prediction matched actual on validation set
                    
                    #Iterate over each row of predictions in valpreds
                    for index, row in valpreds.iterrows():
                        
                        pred_dict = {} #Initialize dictionary with our predictions and weights
                        if class_thresh > 0:
                            valtracker = False #Track if a value would be filled if using a class_thresh value
                        
                        #Add models we've submitted to our ensemble model for finding a final prediction
                        if 'RF' in entry:
                            rf_weight = rf_val_score * valpreds['RF_Conf'][index] #Calculate weight to give model
                            pred_dict[valpreds['RF_Pred'][index]] = rf_weight
                            if class_thresh > 0: #Check if value would be filled if using a class_thresh
                                if valpreds['RF_Conf'][index] > class_thresh:
                                    valtracker = True
                        if 'LSVC' in entry:
                            lsvc_weight = lsvc_val_score * valpreds['LSVC_Conf'][index]
                            if valpreds['LSVC_Pred'][index] in pred_dict: #If predicted class has already been predicted
                                pred_dict[valpreds['LSVC_Pred'][index]] = pred_dict[valpreds['LSVC_Pred'][index]] + lsvc_weight
                            else: #If predicted class has not yet been predicted
                                pred_dict[valpreds['LSVC_Pred'][index]] = lsvc_weight
                            if class_thresh > 0:
                                if valpreds['LSVC_Conf'][index] > class_thresh:
                                    valtracker = True
                        if 'KNN' in entry:
                            knn_weight = knn_val_score * valpreds['KNN_Conf'][index]
                            if valpreds['KNN_Pred'][index] in pred_dict:
                                pred_dict[valpreds['KNN_Pred'][index]] = pred_dict[valpreds['KNN_Pred'][index]] + knn_weight
                            else:
                                pred_dict[valpreds['KNN_Pred'][index]] = knn_weight
                            if class_thresh > 0:
                                if valpreds['KNN_Conf'][index] > class_thresh:
                                    valtracker = True
                        if 'GBC' in entry:
                            gbc_weight = gbc_val_score * valpreds['GBC_Conf'][index]
                            if valpreds['GBC_Pred'][index] in pred_dict:
                                pred_dict[valpreds['GBC_Pred'][index]] = pred_dict[valpreds['GBC_Pred'][index]] + gbc_weight
                            else:
                                pred_dict[valpreds['GBC_Pred'][index]] = gbc_weight
                            if class_thresh > 0:
                                if valpreds['GBC_Conf'][index] > class_thresh:
                                    valtracker = True
                        if 'LR' in entry:
                            lr_weight = lr_val_score * valpreds['LR_Conf'][index]
                            if valpreds['LR_Pred'][index] in pred_dict:
                                pred_dict[valpreds['LR_Pred'][index]] = pred_dict[valpreds['LR_Pred'][index]] + lr_weight
                            else:
                                pred_dict[valpreds['LR_Pred'][index]] = lr_weight
                            if class_thresh > 0:
                                if valpreds['LR_Conf'][index] > class_thresh:
                                    valtracker = True
                        if 'ADA' in entry:
                            ada_weight = ada_val_score * valpreds['ADA_Conf'][index]
                            if valpreds['ADA_Pred'][index] in pred_dict:
                                pred_dict[valpreds['ADA_Pred'][index]] = pred_dict[valpreds['ADA_Pred'][index]] + ada_weight
                            else:
                                pred_dict[valpreds['ADA_Pred'][index]] = ada_weight
                            if class_thresh > 0:
                                if valpreds['ADA_Conf'][index] > class_thresh:
                                    valtracker = True
                        
                        final_pred = max(pred_dict.items(), key=operator.itemgetter(1))[0] #Get dict key with highest value
                        value_list.append(final_pred) #Add prediction with highest weight to our list of predictions
                        if class_thresh > 0:
                            if valtracker == True:
                                val_filled = val_filled + 1
                                if final_pred == valpreds['Actual'][index]:
                                    val_match = val_match + 1
                    
                    valpreds['Final_Pred'] = value_list #Fill our dataframe in with our final predictions
                    
                    #Determine if our final predictions in Final_Pred are better than current best model
                    valmatches = (valpreds['Actual'] == valpreds['Final_Pred']).sum()
                    modelscore = valmatches/(vallength)
                    print('Model Score: ' + str(round(modelscore,4)))
                    
                    if class_thresh > 0: #If we have a threshold on how confident our models are, output more info
                        print('Fraction of rows filled with predictions, given threshold of ' + str(class_thresh) + ' (on validation set):  ' + str(round(val_filled/vallength,4)))
                        if val_filled > 0: #Ensure no division by 0
                            print('Accuracy of predictions:  ' + str(round(val_match/val_filled, 4)))
                    
                    #If model is better than existing best model, make it the new best model
                    if modelscore > bestscore:
                        bestmodel = list(entry)
                        bestscore = modelscore
                        if class_thresh > 0:
                            bestnumfilled = val_filled
                            bestvalmatches = val_match
                        
            #Now use the best combination of models (determined above from the val set) to fill in missing values in test set
            print('Best combination of models or single model: ' + str(bestmodel))
            print('Score for best combination/single model: ' + str(round(bestscore,4)))
            if class_thresh > 0:
                print('Fraction of validation set filled with best model above given threshold of ' + str(class_thresh) + ':  ' + str(round(bestnumfilled/vallength,4)))
                print('Accuracy of filled predictions on validation set above given threshold of ' + str(class_thresh) + ':  ' + str(round(bestvalmatches/bestnumfilled,4)))
            
            print('Filling in missing values with predictions from the above model')
            
            value_list = [] #List containing our model predictions
            
            #Iterate over each row of predictions in allpreds
            for index, row in allpreds.iterrows():
                
                pred_dict = {} #Initialize dictionary with our predictions and weights
                
                if 'RF' in bestmodel:
                    if allpreds['RF_Conf'][index] > class_thresh: #If confidence in prediction is high enough
                        rf_weight = rf_val_score * allpreds['RF_Conf'][index] #Calculate weight to give model
                        pred_dict[allpreds['RF_Pred'][index]] = rf_weight #Add prediction and its weight to dictionary
                if 'LSVC' in bestmodel:
                    if allpreds['LSVC_Conf'][index] > class_thresh:
                        lsvc_weight = lsvc_val_score * allpreds['LSVC_Conf'][index]
                        if allpreds['LSVC_Pred'][index] in pred_dict: #If this prediction is already in dict, give it more weight from lsvc
                            pred_dict[allpreds['LSVC_Pred'][index]] = pred_dict[allpreds['LSVC_Pred'][index]] + lsvc_weight
                        else: #If this prediction isn't already in dict, assign in the weight from lsvc
                            pred_dict[allpreds['LSVC_Pred'][index]] = lsvc_weight
                if 'KNN' in bestmodel:
                    if allpreds['KNN_Conf'][index] > class_thresh:
                        knn_weight = knn_val_score * allpreds['KNN_Conf'][index]
                        if allpreds['KNN_Pred'][index] in pred_dict:
                            pred_dict[allpreds['KNN_Pred'][index]] = pred_dict[allpreds['KNN_Pred'][index]] + knn_weight
                        else:
                            pred_dict[allpreds['KNN_Pred'][index]] = knn_weight
                if 'GBC' in bestmodel:
                    if allpreds['GBC_Conf'][index] > class_thresh:
                        gbc_weight = gbc_val_score * allpreds['GBC_Conf'][index]
                        if allpreds['GBC_Pred'][index] in pred_dict:
                            pred_dict[allpreds['GBC_Pred'][index]] = pred_dict[allpreds['GBC_Pred'][index]] + gbc_weight
                        else:
                            pred_dict[allpreds['GBC_Pred'][index]] = gbc_weight
                if 'LR' in bestmodel:
                    if allpreds['LR_Conf'][index] > class_thresh:
                        lr_weight = lr_val_score * allpreds['LR_Conf'][index]
                        if allpreds['LR_Pred'][index] in pred_dict:
                            pred_dict[allpreds['LR_Pred'][index]] = pred_dict[allpreds['LR_Pred'][index]] + lr_weight
                        else:
                            pred_dict[allpreds['LR_Pred'][index]] = lr_weight
                if 'ADA' in bestmodel:
                    if allpreds['ADA_Conf'][index] > class_thresh:
                        ada_weight = ada_val_score * allpreds['ADA_Conf'][index]
                        if allpreds['ADA_Pred'][index] in pred_dict:
                            pred_dict[allpreds['ADA_Pred'][index]] = pred_dict[allpreds['ADA_Pred'][index]] + ada_weight
                        else:
                            pred_dict[allpreds['ADA_Pred'][index]] = ada_weight
                
                if pred_dict == {}: #If none of the selected models had confidences above our threshold
                    final_pred = np.nan
                else: #If at least once model had confidences above our threshold
                    final_pred = max(pred_dict.items(), key=operator.itemgetter(1))[0] #Get dict key with highest value
                    
                value_list.append(final_pred)
                    
            allpreds['Final_Pred'] = value_list #Fill our dataframe in with our final predictions
                
            if class_thresh > 0:
                print('Number of values filled with confidences above ' + str(class_thresh) + ':  ' + str(allpreds['Final_Pred'].count()))
                print('Number of values not filled (confidences too low):  ' + str(allpreds['Final_Pred'].isna().sum()))
              
        
    else: #If column is numeric or datetime
        #Method:  For numeric variables, our final value will be an average of the results,
        # weighted towards models that were more successful at fitting the data on our validation set.
        # Each prediction is multiplied by its model's validation score.  These are summed.
        # After summing, the total is divided by the sum of each model's validation score to normalize.
        
        print('Best individual model: ' + str(bestmodel[0]))
        print('Best individual model r-squared: ' + str(bestscore))
        
        if len(models) == 1: #If only running with a single model
            if 'RF' in models:
                allpreds['Final_Pred'] = allpreds['RF_Pred']
            if 'LSVC' in models:
                allpreds['Final_Pred'] = allpreds['LSVC_Pred']
            if 'KNN' in models:
                allpreds['Final_Pred'] = allpreds['KNN_Pred']
            if 'GBC' in models:
                allpreds['Final_Pred'] = allpreds['GBC_Pred']
            if 'LR' in models:
                allpreds['Final_Pred'] = allpreds['LR_Pred']
            if 'ADA' in models:
                allpreds['Final_Pred'] = allpreds['ADA_Pred']
        
        if len(models) > 1: #If more than one model was selected
            for i in range(2, len(models) + 1): #Iterate over all combinations of submitted models
                modelslist = list(itertools.combinations(models,i))
                for entry in modelslist:
                    valpreds['Final_Pred'] = 0
                    NumFactor = 0 #Normalizing factor based on method scores
                    print('Testing combination of models: ' + str(entry))
        
                    #Test ensemble performances on validation data
                    if 'RF' in entry:
                        valpreds['Final_Pred'] = valpreds['Final_Pred'] + valpreds['RF_Pred'] * rf_val_score
                        NumFactor = NumFactor + rf_val_score
                    if 'LSVC' in entry:
                        valpreds['Final_Pred'] = valpreds['Final_Pred'] + valpreds['LSVC_Pred'] * lsvc_val_score
                        NumFactor = NumFactor + lsvc_val_score
                    if 'KNN' in entry:
                        valpreds['Final_Pred'] = valpreds['Final_Pred'] + valpreds['KNN_Pred'] * knn_val_score
                        NumFactor = NumFactor + knn_val_score
                    if 'GBC' in entry:
                        valpreds['Final_Pred'] = valpreds['Final_Pred'] + valpreds['GBC_Pred'] * gbc_val_score
                        NumFactor = NumFactor + gbc_val_score
                    if 'LR' in entry:
                        valpreds['Final_Pred'] = valpreds['Final_Pred'] + valpreds['LR_Pred'] * lr_val_score
                        NumFactor = NumFactor + lr_val_score
                    if 'ADA' in entry:
                        valpreds['Final_Pred'] = valpreds['Final_Pred'] + valpreds['ADA_Pred'] * ada_val_score
                        NumFactor = NumFactor + ada_val_score
                        
                    valpreds['Final_Pred'] = valpreds['Final_Pred'] / NumFactor 
        
                    #Check column correlation scores
                    corrval = valpreds['Actual'].corr(valpreds['Final_Pred'])
                    print("Model score: " + str(round(corrval,4)))
                    #Check column R-squared values
                    x = valpreds['Actual']
                    y = valpreds['Final_Pred']
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    r_squared = r_value * r_value
                    print("Model R-squared value: " + str(r_value*r_value))
                    
                    #If model is better than existing best model, make it the new best model
                    #Using the correlation score
                    #if corrval > bestscore:
                    #Using r-squared
                    if r_squared > bestscore:
                        bestmodel = list(entry)
                        #bestscore = corrval #If using correlation score
                        bestscore = r_squared
                        bestcorr = corrval
                        bestvals = valpreds[['Final_Pred']]
                
            
            #Now use the best combination of models (determined above from the val set) to fill in missing values in test set
            valpreds['Final_Pred'] = bestvals
            #Undo scaling transformations on val data before moving on 
            for column in valpreds:
                valpreds[[column]] = yscaler.inverse_transform(valpreds[[column]])
                valpreds[column] = valpreds[column].round(4)
            
            #If column was a datetime, convert back to datetime format
            if column_name in date_fields:
                for column in valpreds:
                    valpreds[column] = valpreds[column].round(0)
                    valpreds[column] = pd.to_datetime(valpreds[column], unit='s')  
                    
            print('Best combination of models or single model: ' + str(bestmodel))
            print('R-squared value for best combination/single model: ' + str(bestscore))
            if len(bestmodel) == 1:  #If best model was only a single model, corr score wasn't calculated yet
                bestcorr = valpreds['Actual'].corr(valpreds['Final_Pred'])
            print('Correlation value for best combination/single model: ' + str(bestcorr))
            print('Filling in missing values with predictions from the above model')
        
            #Use best model from above to make predictions on the test set
            allpreds['Final_Pred'] = 0
            NumFactor = 0 #Normalizing factor based on method scores
            if 'RF' in bestmodel:
                allpreds['Final_Pred'] = allpreds['Final_Pred'] + allpreds['RF_Pred'] * rf_val_score
                NumFactor = NumFactor + rf_val_score
            if 'LSVC' in bestmodel:
                allpreds['Final_Pred'] = allpreds['Final_Pred'] + allpreds['LSVC_Pred'] * lsvc_val_score
                NumFactor = NumFactor + lsvc_val_score
            if 'KNN' in bestmodel:
                allpreds['Final_Pred'] = allpreds['Final_Pred'] + allpreds['KNN_Pred'] * knn_val_score
                NumFactor = NumFactor + knn_val_score
            if 'GBC' in bestmodel:
                allpreds['Final_Pred'] = allpreds['Final_Pred'] + allpreds['GBC_Pred'] * gbc_val_score
                NumFactor = NumFactor + gbc_val_score
            if 'LR' in bestmodel:
                allpreds['Final_Pred'] = allpreds['Final_Pred'] + allpreds['LR_Pred'] * lr_val_score
                NumFactor = NumFactor + lr_val_score
            if 'ADA' in bestmodel:
                allpreds['Final_Pred'] = allpreds['Final_Pred'] + allpreds['ADA_Pred'] * ada_val_score
                NumFactor = NumFactor + ada_val_score
                
            #NumFactor = ModelScores.sum(axis = 1, skipna = True)
            allpreds['Final_Pred'] = allpreds['Final_Pred'] / NumFactor
            
            #Undo scaling transformation we made earlier to speed up learning process
            for column in allpreds:
                allpreds[[column]] = yscaler.inverse_transform(allpreds[[column]])
                allpreds[column] = allpreds[column].round(4)
            
            #If column was a datetime, convert back to datetime format
            if column_name in date_fields:  
                for column in allpreds:
                    allpreds[column] = allpreds[column].round(0)
                    allpreds[column] = pd.to_datetime(allpreds[column], unit='s')        
    
    
    
    ###################################################
    #INPUT PREDICTIONS BACK INTO DATAFRAME
     
    
    #Reload the original dataframe (before we filled NaNs and mangled it in other ways)
    df = cu.load_dataframe(data_folder, data_filename, data_extension)
    #Input our final predictions
    if (column_name in measure_fields) or (column_name in categorical_fields):
        df.loc[test_index,column_name] = allpreds['Final_Pred']
    if column_name in date_fields:  #Date fields need to be input differently to avoid bugs
        for value in test_index:
            df[column_name][value] = allpreds['Final_Pred'][value]
    #Save the dataframe now filled with our predictions for the column
    print('Saving dataframe with predictions entered, denoted FILLED')
    cu.save_file(df, data_folder, data_filename, '_FILLED.csv')
    
    #Save our predictions review (in case we want to see how we made our decisions)
    print('Saving dataframe summarizing method predictions, denoted PREDS')
    #Add index numbers as the first column in the dataframe so that they're present in the saved file
    allpreds['Index'] = allpreds.index
    allpredscols = allpreds.columns.tolist()
    allpredscols.insert(0, allpredscols.pop(allpredscols.index('Index')))
    allpreds = allpreds.reindex(columns=allpredscols)
    cu.save_file(allpreds, data_folder, data_filename, '_PREDS.csv') 
    print('Saving dataframe summarizing method predictions on validation set, denoted VAL')
    cu.save_file(valpreds, data_folder, data_filename, '_VAL.csv')
    
    print('Fit models have been saved in the savedmodels subfolder')
    print(datetime.datetime.now() - start_time)
    start_time = datetime.datetime.now()    
            
    
    
################################################################
#RUN THE COLUMN_CLASSIFICATION FUNCTION
#Default
#column_classification(df, 'TCN Pieces Code')    
    
#Default variant
#column_classification(df = df, column_name = 'ACT_GREATEST_DT', cols_to_use = None, cull_cols = True, models = ['RF', 'LSVC', 'KNN', 'GBC', 'LR', 'ADA'], row_limit = 100000, class_thresh = 0)
    
#Default variant with single list of columns
#column_classification(df = df, column_name = 'CONTRACT_TARIFF', cols_to_use = ['SPOD_ACT_CD', 'VOYDOC_ACT', 'ACT_GREATEST_DTSRC', 
#    'ACT_LEAST_DT', 'ACT_LEAST_DTSRC', 'BK_BOOKER_OFC', 'BK_CONSIGNOR_AGENCY', 'BOOKING_NUM', 'BOOKING_TYPE', 'CONSIGNEE_AGENCY', 
#    'CONSIGNOR_AGENCY', 'CONTAINER_EMPTY_AVAIL_DT', 'DT_CNT', 'EFF_DT', 'EFF_DTSRC', 'EXP_DTSRC', 'FIRST_RDD_DT', 'GATES_RDD_DT', 
#    'GATES_SAIL_DT', 'IBS_HEIGHT', 'IBS_MOVEMENT_TYPE', 'IMPORT_APOD_ACT_ARV_DT', 'IMPORT_APOE_ACT_DPT_DT', 'LOB_SAIL_DT', 'OID', 
#    'PAT_ADJUSTED_RDD_DT', 'SHPMT_PCS', 'SPOD_ACT_ARV_DT', 'SPOD_SCH_ARV_DT', 'SPOE_ACT_DPT_DT', 'SPOE_ACT_INGATE_DT', 
#    'SPOE_ACT_LOAD_DT', 'SPOE_ACT_NM', 'SPOE_SCH_DPT_DT', 'VESSEL_NM_ORIGIN', 'VSTAT'], cull_cols = False, 
#    models = ['RF', 'LSVC', 'KNN', 'GBC', 'LR', 'ADA'], row_limit = 100000, class_thresh = 0.95)

#Classifying a categorical column (with a list of lists for cols_to_use)
#column_classification(df, 'CONTRACT_TARIFF', [['ACT_GREATEST_DT', 'ACT_LEAST_DT', 'LATEST_EVNT_DT'],
#                                              ['ACT_GREATEST_DTSRC', 'ACT_LEAST_DTSRC', 'BK_BOOKER_OFC',
#                 'BOOKING_TYPE', 'CARRIER', 'CONTAINER_OWNERSHIP', 'CONTAINER_SIZE', 'CONTAINER_TYPE',
#                 'FEEDER_DIRECT_INDICATOR', 'IBS_MOVEMENT_TYPE', 'LATEST_EVNT_DESC', 'MOVEMENT_TYPE', 'PREPO_FG', 
#                 'RCVR_AGENCY_NM', 'SHIP_AGENCY_NM', 'SHPMT_STATUS', 'SHPMT_TYPE', 'SUPPLY_CLASS_TXT', 'VESSEL_FLG', 
#                 'VSTAT', 'DIRECTION'],['GATES_HEIGHT', 'GATES_LENGTH', 'GATES_WIDTH', 'IBS_HEIGHT', 'IBS_LENGTH',
#                  'IBS_WIDTH', 'SHPMT_CUBE', 'SHPMT_LBS', 'SHPMT_MTONS', 'SHPMT_PCS', 'SHPMT_TEUS']], cull_cols = False,
#                models = ['RF', 'LSVC', 'KNN', 'GBC', 'LR', 'ADA'], row_limit = 100000, class_thresh = 0.95)

#Classifying a numeric column (with a list of lists for cols_to_use)
#column_classification(df, 'GATES_HEIGHT', [['ACT_GREATEST_DT', 'ACT_LEAST_DT', 'LATEST_EVNT_DT'],
#                                              ['ACT_GREATEST_DTSRC', 'ACT_LEAST_DTSRC', 'BK_BOOKER_OFC',
#                 'BOOKING_TYPE','CARRIER', 'CONTAINER_OWNERSHIP', 'CONTAINER_SIZE', 'CONTAINER_TYPE', 'CONTRACT_TARIFF',
#                 'FEEDER_DIRECT_INDICATOR', 'IBS_MOVEMENT_TYPE', 'LATEST_EVNT_DESC', 'MOVEMENT_TYPE', 'PREPO_FG', 
#                 'RCVR_AGENCY_NM', 'SHIP_AGENCY_NM', 'SHPMT_STATUS', 'SHPMT_TYPE', 'SUPPLY_CLASS_TXT', 'VESSEL_FLG', 
#                 'VSTAT', 'DIRECTION'],['GATES_LENGTH', 'GATES_WIDTH', 'IBS_HEIGHT', 'IBS_LENGTH',
#                  'IBS_WIDTH', 'SHPMT_CUBE', 'SHPMT_LBS', 'SHPMT_MTONS', 'SHPMT_PCS', 'SHPMT_TEUS']], cull_cols = True,
#                models = ['RF', 'LSVC', 'KNN', 'GBC', 'LR', 'ADA'], row_limit = 100000, class_thresh = 0)
    
#Classifying a datetime column (with a list of lists for cols_to_use)
#column_classification(df, 'LATEST_EVNT_DT', [['ACT_GREATEST_DT', 'ACT_LEAST_DT'],
#                                              ['ACT_GREATEST_DTSRC', 'ACT_LEAST_DTSRC', 'BK_BOOKER_OFC',
#                 'BOOKING_TYPE','CARRIER', 'CONTAINER_OWNERSHIP', 'CONTAINER_SIZE', 'CONTAINER_TYPE', 'CONTRACT_TARIFF',
#                 'FEEDER_DIRECT_INDICATOR', 'IBS_MOVEMENT_TYPE', 'LATEST_EVNT_DESC', 'MOVEMENT_TYPE', 'PREPO_FG', 
#                 'RCVR_AGENCY_NM', 'SHIP_AGENCY_NM', 'SHPMT_STATUS', 'SHPMT_TYPE', 'SUPPLY_CLASS_TXT', 'VESSEL_FLG', 
#                 'VSTAT', 'DIRECTION'],['GATES_LENGTH', 'GATES_WIDTH', 'GATES_HEIGHT', 'IBS_HEIGHT', 'IBS_LENGTH',
#                  'IBS_WIDTH', 'SHPMT_CUBE', 'SHPMT_LBS', 'SHPMT_MTONS', 'SHPMT_PCS', 'SHPMT_TEUS']], cull_cols = False)

#Example: Just a list for cols_to_use
#column_classification(df, 'CONTRACT_TARIFF', ['ACT_GREATEST_DTSRC', 'ACT_LEAST_DTSRC', 'BK_BOOKER_OFC',
#                 'BOOKING_TYPE','CARRIER', 'CONTAINER_OWNERSHIP', 'CONTAINER_SIZE', 'CONTAINER_TYPE',
#                 'FEEDER_DIRECT_INDICATOR', 'IBS_MOVEMENT_TYPE', 'LATEST_EVNT_DESC', 'MOVEMENT_TYPE', 'PREPO_FG', 
#                 'RCVR_AGENCY_NM', 'SHIP_AGENCY_NM', 'SHPMT_STATUS', 'SHPMT_TYPE', 'SUPPLY_CLASS_TXT', 'VESSEL_FLG', 
#                 'VSTAT', 'DIRECTION', 'GATES_HEIGHT', 'GATES_LENGTH', 'GATES_WIDTH', 'IBS_HEIGHT', 'IBS_LENGTH',
#                  'IBS_WIDTH', 'SHPMT_CUBE', 'SHPMT_LBS', 'SHPMT_MTONS', 'SHPMT_PCS', 'SHPMT_TEUS'], cull_cols = True)
#
#Classification with full surface dataframe
#column_classification(df, 'CONTRACT_TARIFF', [['ACT_GREATEST_DT','ACT_LEAST_DT', 'BOOKING_RLS_DT','CONSIGNEE_ARV_RFID_DT','CONSIGNEE_ARV_SAT_TAG_DT',
#               'CONTAINER_EMPTY_AVAIL_DT', 'DEST_AVAIL_FOR_DELVRY_DT', 'DEST_DELIVERY_DT', 'DEST_EMPTY_RETURN_DT',
#               'EFF_DT', 'EXP_DT', 'EXPORT_APOD_ACT_ARV_DT', 'EXPORT_APOD_ACT_OUTGATE_DT', 'EXPORT_APOE_ACT_DPT_DT',
#               'EXPORT_APOE_RECEIPT_DT', 'FIRST_RDD_DT', 'GATES_RDD_DT', 'GATES_SAIL_DT', 'IBS_RDD_DT',
#               'IMPORT_APOD_ACT_ARV_DT', 'IMPORT_APOD_ACT_OUTGATE_DT', 'IMPORT_APOE_ACT_DPT_DT', 'IMPORT_APOE_RECEIPT_DT',
#               'LAST_RDD_DT','LATEST_EVNT_DT', 'LOB_CREATE_DT', 'LOB_SAIL_DT', 'MANIFEST_CALL_DT',
#               'PAT_ADJUSTED_RDD_DT', 'PICKUP_DT', 'RFID_LATEST_EVNT_DT', 'SHPMT_DPT_DT', 'SPOD_ACT_ARV_DT',
#               'SPOD_ACT_OUTGATE_DT','SPOD_ACT_UNLOAD_DT', 'SPOD_SCH_ARV_DT', 'SPOE_ACT_DPT_DT',
#               'SPOE_ACT_INGATE_DT', 'SPOE_ACT_LOAD_DT', 'SPOE_SCH_DPT_DT', 'SRFID_LATEST_EVNT_DT',
#               'TCMD_PRINT_DT'],['SHPMT_UNT_ID', 'SPOD_ACT_CD', 'SPOE_ACT_CD', 'VOYDOC_ACT', 'ACT_GREATEST_DTSRC', 'ACT_LEAST_DTSRC',
#                      'BK_BOOKER_OFC','BK_BOOKER_UNIT_NM','BK_CONSIGNEE_AGENCY', 'BK_CONSIGNEE_CD','BK_CONSIGNOR_AGENCY',
#                      'BK_CONSIGNOR_CD','BOOKING_NUM','BOOKING_TYPE','CARRIER','CONSIGNEE_AGENCY','CONSIGNEE_CD',
#                      'CONSIGNOR_AGENCY','CONSIGNOR_CD', 'CONSIGNOR_UIC', 'CONTAINER_NUM', 'CONTAINER_OWNERSHIP',
#                      'CONTAINER_SIZE', 'CONTAINER_TYPE', 'CONTRACT_NUMBER', 'EFF_DTSRC', 'EXP_DTSRC',
#                      'EXPORT_APOD_ACT_LOC_NM', 'EXPORT_APOE_ACT_LOC_NM','FEEDER_DIRECT_INDICATOR', 'FIRST_RDD_DTSRC',
#                      'HANDLING_CD_DESCRIPTION', 'IBS_MOVEMENT_TYPE', 'IBS_TARIFF', 'IMPORT_APOD_ACT_LOC_NM',
#                      'IMPORT_APOE_ACT_LOC_NM', 'LAST_RDD_DTSRC', 'LATEST_EVNT_DESC', 'LATEST_EVNT_LOC',
#                      'MOVEMENT_TYPE', 'NSN', 'PREPO_FG', 'RCVR_AGENCY_NM', 'RFID_LATEST_EVNT_LOC', 'SHIP_AGENCY_NM',
#                      'SHPMT_STATUS', 'SHPMT_TYPE','SPOD_ACT_NM', 'SPOE_ACT_NM', 'SRFID_TAG_ID', 'SUPPLY_CLASS_TXT', 
#                      'TAC', 'TYPE_PACK_CD', 'ULN', 'VESSEL_FLG', 'VESSEL_NM_DEST', 'VESSEL_NM_ORIGIN', 'VESSEL_NM_PRMY', 
#                      'VSTAT', 'WATER_COMMODITY_CD', 'WATER_COMMODITY_TX', 'DIRECTION', 'XLTAB'],['DAYS_DIF', 'DT_CNT', 
#                        'GATES_HEIGHT', 'GATES_LENGTH', 'GATES_WIDTH', 'IBS_HEIGHT', 'IBS_LENGTH',
#                  'IBS_WIDTH', 'OID','PCFN', 'RFID_TAG_ID', 'SHPMT_CUBE', 'SHPMT_LBS', 'SHPMT_MTONS', 'SHPMT_PCS',
#                  'SHPMT_ROW_ID', 'SHPMT_TEUS', 'SRFID_LATEST_LAT_COORD', 'SRFID_LATEST_LONG_COORD' ,'SU_GRP_ID', 
#                  'SU_ROW_ID']], cull_cols = False, models = ['RF', 'LSVC', 'KNN', 'GBC', 'LR', 'ADA'], row_limit = 100000)
#
#Datetime fill with relevant columns
#column_classification(df, 'CONTRACT_TARIFF', [['ACT_LEAST_DT', 'EFF_DT', 'EXP_DT',
#                                              'FIRST_RDD_DT', 'GATES_RDD_DT', 'GATES_SAIL_DT', 'LATEST_EVNT_DT', 'SPOD_ACT_ARV_DT',
#                                              'SPOD_ACT_OUTGATE_DT', 'SPOD_ACT_UNLOAD_DT', 'SPOD_SCH_ARV_DT', 'SPOE_ACT_DPT_DT', 
#                                              'SPOE_ACT_INGATE_DT', 'SPOE_ACT_LOAD_DT', 'SPOE_SCH_DPT_DT'], 
#    ['SHPMT_STATUS', 'LATEST_EVNT_DESC', 'EXP_DTSRC', 'CONSIGNEE_AGENCY'], ['SHPMT_CUBE', 'SHPMT_LBS']], 
#    cull_cols = False, models = ['RF', 'LR'], row_limit = 100000)
#
#SDDB Data for EDD in AFRICOM
#column_classification(df, 'DAYS_ACT_LRT_NOBK', cull_cols = True)
#column_classification(df, 'DAYS_ACT_LRT_NOBK', [['DT_DOC', 'DT_RDD_RQSN'], 
#                                                ['CSGNE_DODAAC', 'CSGNE_Loc', 'DEPOT_ORG_ID', 'DEPOT_SHORT_NM', 'POE_POD_RT_NM', 'RQSN_FSC_CD',
#                                                 'RQSN_Item_Code', 'RQSN_ORG_ID', 'RQSN_RDD_TX', 'RQSN_SCMC', 'RQSN_SOS_ORG_ID',
#                                                 'RQSN_Supply_Priority', 'STRAT_SHIPMODE_CD', 'STREAM_CUSTOMER', 'STREAM_DEPOT_GROUP',
#                                                 'STREAM_NM', 'TransMode', 'DEPOT_SUPL_MGD', 'RQSN_SCMC_DESCRPTN', 'STRAT_MVMT_METHOD'], 
#                                                ['DAYS_STD_LRT', 'DAYS_STD_MRO_WHSEs', 'DAYS_STD_Source', 'DAYS_STD_Supplier', 
#                                                 'DAYS_STD_Theater', 'DAYS_STD_Transporter', 'ONG_LRT', 'RQSN_QTY_NUMBER',
#                                                 'RQSN_SHIPWT', 'RQSN_SQTY_AMT', 'RQSN_UNIT_WEIGHT', 'RQSN_UNIT_PRICE']]) #Top 34 columns
#column_classification(df, 'DAYS_ACT_LRT_NOBK', [['DT_DOC', 'DT_RDD_RQSN'],
#                                                ['DEPOT_ORG_ID', 'RQSN_RDD_TX', 'RQSN_SCMC', 'STREAM_DEPOT_GROUP', 'STREAM_NM',
#                                                 'DEPOT_SUPL_MGD', 'RQSN_SCMC_DESCRPTN', 'STRAT_MVMT_METHOD'],
#                                                ['DAYS_STD_LRT', 'DAYS_STD_Supplier', 'DAYS_STD_Transporter', 'ONG_LRT', 'RQSN_SHIPWT',
#                                                 'RQSN_UNIT_WEIGHT']], cull_cols = True) #Top 16 columns
#column_classification(df, 'DAYS_ACT_LRT_NOBK', [[],
#                                                ['DEPOT_ORG_ID', 'RQSN_RDD_TX', 'STREAM_DEPOT_GROUP', 'STREAM_NM',
#                                                 'DEPOT_SUPL_MGD', 'RQSN_SCMC_DESCRPTN', 'RQSN_SCMC', 'STRAT_MVMT_METHOD'],
#                                                ['DAYS_STD_LRT', 'DAYS_STD_Supplier', 'DAYS_STD_Transporter', 'ONG_LRT', 'RQSN_SHIPWT',
#                                                 'RQSN_UNIT_WEIGHT']]) #No dates
#column_classification(df, 'DAYS_ACT_LRT_NOBK', [[],  
#                                                ['CSGNE_DODAAC', 'CSGNE_Loc', 'DEPOT_ORG_ID', 'DEPOT_SHORT_NM', 'POE_POD_RT_NM', 'RQSN_FSC_CD',
#                                                 'RQSN_Item_Code', 'RQSN_ORG_ID', 'RQSN_RDD_TX', 'RQSN_SCMC', 'RQSN_SOS_ORG_ID',
#                                                 'RQSN_Supply_Priority', 'STRAT_SHIPMODE_CD', 'STREAM_CUSTOMER', 'STREAM_DEPOT_GROUP',
#                                                 'STREAM_NM', 'TransMode', 'DEPOT_SUPL_MGD', 'RQSN_SCMC_DESCRPTN', 'STRAT_MVMT_METHOD'], 
#                                                ['ONG_LRT', 'RQSN_QTY_NUMBER',
#                                                 'RQSN_SHIPWT', 'RQSN_SQTY_AMT', 'RQSN_UNIT_WEIGHT', 'RQSN_UNIT_PRICE']])  #No dates/standards
#column_classification(df, 'DAYS_ACT_LRT_NOBK', [[],
#                                                ['CSGNE_Loc', 'DEPOT_ORG_ID', 'POE_POD_RT_NM', 'RQSN_Item_Code', 'RQSN_ORG_ID',
#                                                 'RQSN_RDD_TX', 'RQSN_SCMC', 'RQSN_SOS_ORG_ID', 'STREAM_NM', 
#                                                 'DEPOT_SUPL_MGD', 'STRAT_MVMT_METHOD'],
#                                                ['RQSN_SHIPWT']]) #Top 12 columns with no dates/standards
#
#
#SDDB Data for EDD in EUCOM
#column_classification(df, 'DAYS_ACT_LRT_NOBK', [['DT_DOC', 'DT_EST', 'DT_FILEDATE', 'DT_MRO', 'DT_RDD_RQSN', 'DT_RTN'],
#                                                ['AFLOAT_FG', 'CSGNE_CCMD', 'CSGNE_Country', 'CSGNE_DODAAC', 'CSGNE_Loc', 'CSGNE_ORG_NM',
#                                                 'CSGNE_REGN_Std', 'CSGNE_SVC_GRP', 'DEPOT_AOR_Group', 'DEPOT_CCMD', 'DEPOT_CITY',
#                                                 'DEPOT_GENC_CTRY_CD', 'DEPOT_GROUPING', 'DEPOT_SHORT_NM', 'DOCSUFFVAL',
#                                                 'ICP_NM', 'ICP_ORG_ID', 'ICP_ORG_ID_TYPE', 'ICP_SUPL_MGD', 'ONG_CSGNE_REGN', 'ONG_Nm',
#                                                 'POE_POD_RT_NM', 'RQSN_FSC_CD', 'RQSN_IPG', 'RQSN_Item_Code_Type',
#                                                 'RQSN_ORG_ID', 'RQSN_RDD_TX', 'RQSN_SCMC', 'RQSN_Service', 
#                                                 'RQSN_Service_Group', 'RQSN_SIGNAL_CD', 'RQSN_SOS_ORG_ID',
#                                                 'RQSN_Supply_Priority', 'RQSN_TDD_CAT', 'SDP_M_REGIONS', 'STRAT_Carrier_Dropoff', 
#                                                 'STRAT_SHIPMODE_CD', 'STRAT_ShipMode_NM', 'STREAM_ADDITIVE', 'STREAM_CUSTOMER',
#                                                 'STREAM_DEPOT_GROUP', 'STREAM_NM', 'STREAM_STRAT_TRANS_METHOD', 'TransMode',
#                                                 'CSGNE_Loc_Src', 'CSGNE_ORG_ID_SRC', 'CSGNE_SVC_NM', 'DEPOT_GRP_Src', 'DEPOT_SUPL_MGD',
#                                                 'DEPOT_SUPL_OWNED', 'DTSrc_EST', 'DTSrc_MRO', 'DTSrc_RTN', 'RQSN_FSC_DESCRPTN',
#                                                 'RQSN_SC_DESCRPTN', 'RQSN_SCMC_DESCRPTN'],
#                                                ['DAYS_STD_AVAIL_MRA', 'DAYS_STD_DOC_EST', 'DAYS_STD_EST_RTN', 'DAYS_STD_LRT', 'DAYS_STD_MRO_WHSEs',
#                                                'DAYS_STD_RTN_MRO_NOBK', 'DAYS_STD_Source', 'DAYS_STD_Supplier', 'DAYS_STD_Theater', 'ONG_LRT',
#                                                'RQSN_QTY_NUMBER', 'RQSN_SHIPWT', 'RQSN_SQTY_AMT', 'RQSN_UNIT_WEIGHT', 'NONBK_CALC', 'RQSN_UNIT_PRICE']],
#                                                 cull_cols = True) 
#column_classification(df, 'DAYS_ACT_LRT_NOBK', [['DT_DOC', 'DT_EST', 'DT_FILEDATE', 'DT_MRO', 'DT_RDD_RQSN', 'DT_RTN'],
#                                                ['CSGNE_Country', 'CSGNE_DODAAC', 'CSGNE_Loc', 'CSGNE_ORG_NM',
#                                                 'DEPOT_CITY', 'DEPOT_SHORT_NM', 'DOCSUFFVAL', 'ICP_SUPL_MGD', 'ONG_Nm',
#                                                 'POE_POD_RT_NM', 'RQSN_FSC_CD', 'RQSN_ORG_ID', 'RQSN_RDD_TX', 'RQSN_SCMC', 'RQSN_SIGNAL_CD', 
#                                                 'RQSN_Supply_Priority', 'STRAT_Carrier_Dropoff', 'STRAT_SHIPMODE_CD', 'STRAT_ShipMode_NM', 
#                                                 'STREAM_NM', 'DEPOT_SUPL_OWNED', 'RQSN_FSC_DESCRPTN', 'RQSN_SCMC_DESCRPTN'],
#                                                ['DAYS_STD_LRT', 'DAYS_STD_MRO_WHSEs', 'DAYS_STD_Supplier', 'DAYS_STD_Theater', 'ONG_LRT',
#                                                'RQSN_QTY_NUMBER', 'RQSN_SHIPWT', 'RQSN_UNIT_WEIGHT', 'RQSN_UNIT_PRICE']])
#
#
#6-9 months of Ocean data
#column_classification(df, 'DAYS_ACT_LRT_NOBK', [['DT_DOC'],
#                                                ['CCP_DODAAC', 'CCP_State', 'CMDTY_CD', 'CSGNE_DODAAC', 'CSGNE_GENC_CTRY_CD',
#                                                 'CSGNE_ORG_NM', 'CSGNE_REGN_Std', 'DEPOT_CITY', 'DEPOT_State', 'ONG_Nm', 'POE_POD_RT_NM',
#                                                 'RQSN_FSC_CD', 'RQSN_Item_Code', 'RQSN_ORG_ID', 'RQSN_SCMC', 'RQSN_SIGNAL_CD',
#                                                 'RQSN_Supply_Priority', 'STRAT_CARRIER_SCAC', 'STRAT_POD_CTRY_NM',
#                                                 'STRAT_POD_LOC_ID', 'STRAT_POE_LOC', 'STRAT_SHIPMODE_CD', 
#                                                 'STREAM_CUSTOMER', 'CSGNE_SVC_NM'],
#                                                ['DAYS_STD_LRT', 'DAYS_STD_MRO_WHSEs', 'DAYS_STD_POEs_PODr', 'DAYS_STD_Source', 
#                                                 'DAYS_STD_Supplier', 'DAYS_STD_Theater', 'DAYS_STD_Transporter', 'ONG_LRT', 
#                                                 'RQSN_QTY_NUMBER', 'RQSN_SHIPWT', 'RQSN_SQTY_AMT', 'RQSN_UNIT_WEIGHT', 
#                                                 'RQSN_UNIT_PRICE']], row_limit = 500000)
#column_classification(df, 'DAYS_ACT_LRT_NOBK', [['DT_DOC', 'DT_RDD_RQSN'],
#                                                ['AFLOAT_FG', 'CNTR_TYPE', 'CSGNE_CCMD', 'CSGNE_GENC_CTRY_CD', 'CSGNE_REGN_Std', 'CSGNE_State',
#                                                 'CSGNE_SVC_GRP', 'CSGNE_SVC_NM', 'DOCSUFFVAL', 'SDP_M_REGIONS', 'STREAM_CUSTOMER'],
#                                                 []], row_limit = 200000) #Basic 13 columns
#column_classification(df, 'DAYS_ACT_LRT_NOBK', [['DT_DOC', 'DT_RDD_RQSN'],
#                                                ['AFLOAT_FG', 'CNTR_TYPE', 'CSGNE_CCMD', 'CSGNE_GENC_CTRY_CD', 'CSGNE_REGN_Std', 'CSGNE_State',
#                                                 'CSGNE_SVC_GRP', 'CSGNE_SVC_NM', 'DOCSUFFVAL', 'SDP_M_REGIONS', 'STREAM_CUSTOMER',
#                                                 'RQSN_IPG', 'RQSN_Item_Code_Type', 'RQSN_SCMC', 'RQSN_SC_DESCRPTN', 'RQSN_SCMC_DESCRPTN',
#                                                 'RQSN_Service', 'RQSN_Service_Group', 'RQSN_SIGNAL_CD', 'RQSN_SOS_ORG_ID',
#                                                 'RQSN_Supply_Priority', 'RQSN_TDD_CAT'],
#                                                 ['RQSN_QTY_NUMBER', 'RQSN_RDD_TX', 'RQSN_SHIPWT', 'RQSN_SQTY_AMT',
#                                                  'RQSN_UNIT_PRICE', 'RQSN_UNIT_WEIGHT']]) #Basic 13 columns + RQSN columns

#Testing performance on 6 different COCOMS
#column_classification(df, 'DAYS_ACT_LRT_NOBK', [['DT_DOC', 'DT_RDD_RQSN'],
#                                                ['AFLOAT_FG', 'CNTR_TYPE', 'CSGNE_CCMD', 'CSGNE_GENC_CTRY_CD', 'CSGNE_REGN_Std', 'CSGNE_State',
#                                                 'CSGNE_SVC_GRP', 'CSGNE_SVC_NM', 'DOCSUFFVAL', 'SDP_M_REGIONS', 'STREAM_CUSTOMER'],
#                                                 []], row_limit = 800000) #Basic 13 columns
column_classification(df, 'DAYS_ACT_LRT_NOBK', [['DT_DOC', 'DT_RDD_RQSN'],
                                                ['AFLOAT_FG', 'CNTR_TYPE', 'CSGNE_CCMD', 'CSGNE_GENC_CTRY_CD', 'CSGNE_REGN_Std', 'CSGNE_State',
                                                 'CSGNE_SVC_GRP', 'CSGNE_SVC_NM', 'DOCSUFFVAL', 'SDP_M_REGIONS', 'STREAM_CUSTOMER',
                                                 'RQSN_IPG', 'RQSN_Item_Code_Type', 'RQSN_SCMC', 'RQSN_SC_DESCRPTN', 'RQSN_SCMC_DESCRPTN',
                                                 'RQSN_Service', 'RQSN_Service_Group', 'RQSN_SIGNAL_CD', 'RQSN_SOS_ORG_ID',
                                                 'RQSN_Supply_Priority', 'RQSN_TDD_CAT'],
                                                 ['RQSN_QTY_NUMBER', 'RQSN_RDD_TX', 'RQSN_SHIPWT', 'RQSN_SQTY_AMT',
                                                  'RQSN_UNIT_PRICE', 'RQSN_UNIT_WEIGHT']],
                                                 models = ['RF', 'LSVC', 'KNN', 'GBC', 'LR', 'ADA'], row_limit = 800000) #Basic 13 columns + RQSN columns
    