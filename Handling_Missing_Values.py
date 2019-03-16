import pandas as pd


def InfoAboutDataset(column_names,dataframe):
    '''This function prints the information about the data'''

    print ('\n\n')
    print (dataframe.describe())
    print ('\n\n')
    print (dataframe.info())


def ReadDatasetFromFile():
    ''' This function reads the dataset from local directory'''

    filename = "/Users/tejasvibelsare/Library/Mobile Documents/com~apple~CloudDocs/Spring 2019/Data Mining/MiniProject1/adult 2.txt"
    column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
                    "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "classification"]
    dataframe = pd.read_csv(filename, header = None, names = column_names, sep = ',\s', na_values = ["?"],engine='python')
    return (dataframe,column_names)


def MissingValuesOfWorkclass(dataframe):
    ''' This function replaces the missing value of workclass attribute with the mode value of the data'''

    Mode = dataframe.loc[:, "workclass"].mode()     ##### getting mode
    dataframe = dataframe.fillna({"workclass":"Private"})        ### filled misisng values with most frequently occuring value in data
    return dataframe


def MissingValuesOfOccupation(dataframe):
    ''' This function replaces the missing value of occupation attribute with the mode value of the data'''

    occupation_mode = dataframe.loc[:,"occupation"].mode()
    dataframe = dataframe.fillna({"occupation": "Prof-specialty"})  ### filled misisng values with most frequently occuring value in data
    return dataframe


def MissingValuesOfNativeCountry(dataframe):
    ''' This function replaces the missing value of native-country attribute with the mode value of the data'''

    native_country_mode = dataframe.loc[:, "native-country"].mode()
    dataframe = dataframe.fillna({"native-country": "United-States"})  ### filled misisng values with most frequently occuring value in data
    return dataframe


def MissingValuesByMode():
    ''' It checks for missing values in attributes and calls functions to fill them'''

    dataframe, column_names = ReadDatasetFromFile()
    InfoAboutDataset(column_names, dataframe)
    dataframe = dataframe.dropna(how='all')  ##### it will drop the rows only if all values in a row are missing
    dataframe = MissingValuesOfWorkclass(dataframe)
    dataframe = MissingValuesOfOccupation(dataframe)
    dataframe = MissingValuesOfNativeCountry(dataframe)
    return dataframe, column_names

def MissingValuesByRemoving():
    ''' This function removes the records having one or more missing values'''

    dataframe, column_names = ReadDatasetFromFile()
    dataframe = dataframe.dropna(how='any')
    InfoAboutDataset(column_names, dataframe)
    return dataframe, column_names