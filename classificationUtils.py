import pandas as pd
import numpy as np
import datetime
import os


#Load a data file given it's location, name, and extension
def load_dataframe(file_path, file_name, file_type):
    
    #read the file into a dataframe
    try: #first try pipe-separated variables
        df=pd.read_csv(file_path + file_name + file_type, sep="|", encoding='latin1', low_memory=False) #low_memory suppresses dtype error
        print('Loaded dataframe as pipe-separated variables.')
    except:
        try: #second try tab-separated variables
            df=pd.read_csv(file_path + file_name + file_type, sep="\t", encoding='latin1', low_memory=False)
            print('Loaded dataframe as tab-separated variables.')
        except:
            try: #third try comma-separated variables
                df=pd.read_csv(file_path + file_name + file_type, sep=",", encoding='latin1', low_memory=False)
                print('Loaded dataframe as comma-separated variables.')
            except:
                print('Unable to load file as pipe, tab, or comma-separated.')
    
    return(df)
  
    
#Save a file (probably after cleaning)
def save_file(df, file_path, file_name, addon="_FILLED.csv"):
    #Save Clean Dataset
    print("Saving Data...")
    df.to_csv(file_path + file_name + addon, sep="|", index = False) #Use pipe seps
    print("File Saved Successfully")
    
    
def check_directory(directory):
    #Check to see if directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
def clean_date_data(df, fields):
    
    #Loop through the fields
    for field in fields:
        
        #Make sure the field is in the dataframe
        if field in df.columns:
        
            #If it is already a date, then skip it
            if field not in df.select_dtypes(include=['datetime','datetime64','datetime64[ns]','<M8[ns]']).columns:
                
                print(field)

                #Track memory usage #NOTE:  Simply tracking memory usage causes python to reduce memory usage
                MemUsage = df.memory_usage().sum()
                
                #Convert to string
                df[field] = df[field].astype(str)

                #Remove_Spaces
                df[field] = df[field].astype(str).str.strip()

                #Replace invalid entries with nulls
                df[field] = df[field].replace("N/A",pd.NaT)
                df[field] = df[field].replace("NaT",pd.NaT)
                df[field] = df[field].replace("nan",pd.NaT)
                df[field] = df[field].replace("",pd.NaT)
                
                #Blank out fields that contain characters
                regexp = "[a-zA-Z]+"
                df[field] = np.where(df[field].str.match(regexp),pd.NaT,df[field])

                #Attempt conversions with different date formats; using the right format is much faster for to_datetime
                #Test first 1000 entries
                testdf = df[field].dropna().head(1000)
                dcthresh = 0.9 #Threshold for success in datetime conversion formattings
                #If first 1000 nonnull entries agree with one formatting above dcthresh, use it for the column
                if len(pd.to_datetime(testdf, format="%m/%d/%Y", errors='coerce').dropna())/len(testdf) >= dcthresh:
                    df[field] = pd.to_datetime(df[field], format="%m/%d/%Y", errors='coerce')
                elif len(pd.to_datetime(testdf, format="%m/%d/%Y %H:%M:%S.%f", errors='coerce').dropna())/len(testdf) >= dcthresh:
                    df[field] = pd.to_datetime(df[field], format="%m/%d/%Y %H:%M:%S.%f", errors='coerce')
                elif len(pd.to_datetime(testdf, format="%m-%d-%Y", errors='coerce').dropna())/len(testdf) >= dcthresh:
                    df[field] = pd.to_datetime(df[field], format="%m-%d-%Y", errors='coerce')
                elif len(pd.to_datetime(testdf, format="%m-%d-%Y %H:%M:%S.%f", errors='coerce').dropna())/len(testdf) >= dcthresh:
                    df[field] = pd.to_datetime(df[field], format="%m-%d-%Y %H:%M:%S.%f", errors='coerce')
                #Try inferring datetime format, if none of this works use the guaranteed but slow method 
                elif len(pd.to_datetime(testdf, infer_datetime_format = True, errors='coerce').dropna())/len(testdf) >= dcthresh:
                    df[field] = pd.to_datetime(df[field], infer_datetime_format = True, errors='coerce')
                else: #Slow but will almost always work
                    df[field] = pd.to_datetime(df[field], errors='coerce')
                df[field] = pd.to_datetime(df[field], errors='coerce') 
                
                #Remove dates that are in the future (these probably don't make sense)
                df[field] = df[field].apply(lambda x: x if x <= datetime.datetime.now() else pd.NaT)
                
                #Update memory usage
                MemUsage = df.memory_usage().sum()
        
    return(df)


def clean_categorical_data(df, fields):
    
    #Loop through the fields
    for field in fields:
        
        print(field)
        
        #Check to make sure the field is in the dataframe
        if field in df.columns:

            #Track memory usage  #NOTE:  Simply tracking memory usage causes python to reduce memory usage
            MemUsage = df.memory_usage().sum()
            
            #Remove_Spaces
            df[field] = df[field].astype(str).str.strip()

            #Set to null if it contains only a questionmark
            regexp = "^\\?$"
            df[field] = np.where(df[field].str.match(regexp),np.nan,df[field])

            regexp = "[0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{1,4}"
            df[field] = np.where(df[field].str.match(regexp),np.nan,df[field])

            #Replace invalid entries with nulls
            df[field] = df[field].replace("N/A",np.nan)
            df[field] = df[field].replace("nan",np.nan)
            df[field] = df[field].replace("NaN",np.nan)
            df[field] = df[field].replace("",np.nan)
            df[field] = df[field].replace("UNKNOWN",np.nan) ##Added for CONTRACT_TARIFF
            #df[field] = df[field].replace("UNK", np.nan) ##Added as another possibility for NaN
            
            #Convert to category data types if the dtype is object and there are less 50% unique values
            if df[field].nunique() < (len(df)*0.5):

                #Convert the column to a category (this saves lots of memory)
                df[field] = df[field].astype('category')
                
            #Update memory usage
            MemUsage = df.memory_usage().sum() 

    return(df)


def clean_measure_data(df, fields):
    
    for field in fields:
        
        print(field)
        
        #Check to make sure the field is in the dataframe
        if field in df.columns:

            #Track memory usage  #NOTE:  Simply tracking memory usage causes python to reduce memory usage
            MemUsage = df.memory_usage().sum()
            
            #Remove_Spaces
            df[field] = df[field].astype(str).str.strip()
            
            #Remove $ signs and commas (common in dataframe columns)
            df[field] = df[field].str.replace(',','')
            df[field] = df[field].str.replace('$', '')

            regexp = "^\\?$"
            df[field] = np.where(df[field].str.match(regexp),np.nan,df[field])

            regexp = "[0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{1,4}"
            df[field] = np.where(df[field].str.match(regexp),np.nan,df[field])

            regexp = "^[ ]*[-]?[ ]*[0-9]*[.]?[0-9]+[ ]*$"
            df[field] = np.where(df[field].str.match(regexp),df[field], np.nan)

            #Replace invalid entries with nulls
            df[field] = df[field].replace("N/A",np.nan)
            df[field] = df[field].replace("nan",np.nan)
            df[field] = df[field].replace("NaN",np.nan)
            df[field] = df[field].replace("",np.nan)

            #Convert it back into a number
            df[field] = pd.to_numeric(df[field])
            
            #Update memory usage
            MemUsage = df.memory_usage().sum()
        
    return(df)
    
    
def determine_dtypes(df):
    
    measure_fields = []
    date_fields = []
    categorical_fields = []
    nan_fields = []

    #Loop over columns, add columns to correct list (measure/date/categorical fields)
    rowcheck_threshold = 2500 #Set number of rows to investigate when determining column data type ##
    for column in df:
        #Check if column is entirely filled with NaN's.  If so, no point cleaning
        if pd.isnull(df[column]).sum() == len(df[column]):
            print('Column is entirely NaN, excluding from analysis: ' + str(column))
            nan_fields.append(column)
        #Check if column is numerical ('measure' field)
        if (np.issubdtype(df[column].dtype, np.number) == True) and (column not in nan_fields):
            measure_fields.append(column)
            print('Column is numerical: ' + str(column))

        #Check if column is numerical but stored as strings (or has a few string typos that coverted column type to object)
        if column not in nan_fields and column not in measure_fields:
            numsuccess = 0 #Used to track number of successful numerical checks
            numfail = 0 #Used to track number of unsuccessful numerical checks
            for index, row in df.head(rowcheck_threshold).iterrows():
                #We will check first n rows (set by rowcheck_threshold) and make sure they are 99% consistent with numerical values
                if pd.isnull(df[column][index]) == False:
                    try:
                        df[column][index].isdigit() == True #Try to use .isdigit() ('1' == True, 1 -> error, 'One' -> False)
                    except:
                        numsuccess = numsuccess + 1  #.isdigit() will fail for ints/floats
                    else:
                        if df[column][index].isdigit() == True:
                            numsuccess = numsuccess + 1 
                        else:
                            numfail = numfail + 1
            #Check to see that 99%+ of the column values return numerical values        
            if numsuccess != 0 or numfail != 0: #Avoid division by 0
                if (numsuccess/(numsuccess + numfail)) >= 0.99:
                    measure_fields.append(column)
                    print('Column is numerical: ' + str(column))

        #Now determine if column is datetime or categorical
        if column not in nan_fields and column not in measure_fields:
            datesuccess = 0 #Used to track number of successful datetime conversions
            datefail = 0 #Used to track number of unsuccessful datetime conversions
            for index, row in df.head(rowcheck_threshold).iterrows():
                #We will check first n rows (set by rowcheck_threshold) and make sure they are 90% consistent with datetime standards
                if pd.isnull(df[column][index]) == False: #Only test non-null values
                    try:
                        pd.to_datetime(df[column][index]) #See if value can be converted to datetime
                    except:
                        datefail = datefail + 1 #If conversion fails, increment datefail
                    else:
                        #Make sure date makes sense; we shouldn't have pre-1990 data
                        if pd.to_datetime(df[column][index]) > pd.to_datetime('1990-01-01'): 
                            datesuccess = datesuccess + 1
                        else:
                            datefail = datefail + 1 #If date is pre 1990, doesn't make sense (ex: '1' -> '1970-01-01')
            #Check to see that 90%+ of the column values return good datetimes
            if datesuccess != 0 or datefail != 0: #Avoid division by 0; this only happens if first n entries (set by rowcheck_threshold) are NaN
                if (datesuccess/(datesuccess + datefail)) >= 0.9:
                    date_fields.append(column)
                    print('Column is datetime: ' + str(column))
                else:
                    categorical_fields.append(column)
                    print('Column is categorical: ' + str(column))
            else: #If first n rows are all NaN (datesuccess and datefail == 0), exclude column from analysis
                print('All tested entries are NaN, excluding column from analysis: ' + str(column))
                del df[column]
    
    #Ignore columns that are entirely null
    for column in df:
        if column in nan_fields: #Exclude fully NaN columns
            print('Excluding fully NaN columns from analysis: ' + str(column))
            del df[column]
        
    return(df, measure_fields, date_fields, categorical_fields)


#Label encoding class to allow for unknown variables to be encountered
from sklearn.preprocessing import LabelEncoder

class LabelEncoderExt(object):
    def __init__(self):
        #This differs from labelEncoder by handling new classes and providing a value for them [Unknown]
        # Unknown will be added in fit and transform will take care of the new item.  It gives unknown class id
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_
        
    def fit(self, data_list):
        #This will fit the encoder for all the unique values and introduce the Unknown value
        # data_list:  A list of strings
        # returns self
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_
        
        return self
    
    def transform(self, data_list):
        #This will transform the data_list to id list where the new values get assigned to Unknown class
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]
                
        return self.label_encoder.transform(new_data_list)
        
    def inverse_transform(self, data_list):
        #This will perform the inverse transform function
        inverse_data_list = list(data_list)
        return self.label_encoder.inverse_transform(inverse_data_list)