import Handling_Missing_Values
import pandas as pd
import numpy as np
import math
import scipy.stats
# import time

def SplittingDataset(dataframe):
    ''' This function splits the complete dataset in 10 folds'''

    list_of_splits=[]
    number_of_records_in_each_split = int(len(dataframe)/10)

    for i in range(10):
        list_of_splits.append(dataframe[i * number_of_records_in_each_split : (i+1) * number_of_records_in_each_split])
    list_of_splits.append(dataframe[9 * number_of_records_in_each_split:])
    return list_of_splits


def CountOfContinuousAttrTraining(list_of_splits,testing_set_num):
    ''' This function works for converting continuous variables to discrete bins. Gaussian distribution and Equal width binning'''

    bin_wise_count_greater_than_50, bin_wise_count_less_than_50, continuous_attributes_bins= EqualWidthBinning(list_of_splits,testing_set_num)
    return bin_wise_count_greater_than_50, bin_wise_count_less_than_50, continuous_attributes_bins


def EqualWidthBinning(list_of_splits,testing_set_num):
    ''' This function calculate bins by equal binning method to convert continuous variables to discrete bins'''

    bin_wise_count_greater_than_50 = {}
    bin_wise_count_less_than_50 = {}
    continuous_attributes_bins = ['age_bins','capital_gain_bins','capital_loss_bins','education_num_bins','fnlwgt_bins','hours_per_week_bins']
    age_bins = [0, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, np.inf]
    age_names = [0, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90]
    capital_gain_bins = [0, 7000, 14000, 21000, 28000, 35000, 42000, 49000, 56000, 63000, 70000, 77000, 84000, 91000, 98000, np.inf]
    capital_gain_names = [0, 7000, 14000, 21000, 28000, 35000, 42000, 49000, 56000, 63000, 70000, 77000, 84000, 91000, 98000]
    capital_loss_bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, np.inf]
    capital_loss_names = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    education_num_bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, np.inf]
    education_num_names = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    fnlwgt_bins = [0, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 240000, 270000, 300000, 330000, 360000, 390000, 420000, 450000, 480000, 510000,
            540000, 570000, 600000, 630000, 660000, 690000, 720000, 750000, 780000,np.inf]
    fnlwgt_names = [0, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 240000, 270000, 300000, 330000, 360000, 390000,420000, 450000, 480000, 510000,
             540000, 570000, 600000, 630000, 660000, 690000, 720000, 750000, 780000]
    hours_per_week_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
    hours_per_week_names = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


    for i in range(10):
        if i == testing_set_num:
            continue
        else:
            data = list_of_splits[i]
            data['age_bins'] = pd.cut(data['age'], age_bins, labels=age_names)
            data['capital_gain_bins'] = pd.cut(data['capital-gain'], capital_gain_bins, labels=capital_gain_names)
            data['capital_loss_bins'] = pd.cut(data['capital-loss'], capital_loss_bins, labels=capital_loss_names)
            data['education_num_bins'] = pd.cut(data['education-num'], education_num_bins, labels=education_num_names)
            data['fnlwgt_bins'] = pd.cut(data['fnlwgt'], fnlwgt_bins, labels=fnlwgt_names)
            data['hours_per_week_bins'] = pd.cut(data['hours-per-week'], hours_per_week_bins, labels=hours_per_week_names)

            for every_column in continuous_attributes_bins:

                list_of_unique_value_counts = data[every_column].value_counts()
                for unique_index, everyUnique in list_of_unique_value_counts.iteritems():
                    count_less_than = len(data.loc[(data[every_column] == unique_index) & (data["classification"] == "<=50K"), [every_column,"classification"]])

                    if every_column not in bin_wise_count_less_than_50:  #### if column is not present in dictionary as key
                        series = pd.Series([count_less_than], index=[unique_index])
                        bin_wise_count_less_than_50[every_column] = series

                    else:  #### if column is present in dictionary as key
                        if unique_index in (bin_wise_count_less_than_50[every_column].index):  ##### if index is present
                            bin_wise_count_less_than_50[every_column][unique_index] += count_less_than

                        else:  ##### if index is not present
                            bin_wise_count_less_than_50[every_column][unique_index] = count_less_than
                  ################################# greater thn 50

                    count_greater_than = len(data.loc[(data[every_column] == unique_index) & (data["classification"] == ">50K"), [every_column,"classification"]])

                    if every_column not in bin_wise_count_greater_than_50:  #### if column is not present in dictionary as key
                        series = pd.Series([count_greater_than], index=[unique_index])
                        bin_wise_count_greater_than_50[every_column] = series

                    else:  #### if column is present in dictionary as key
                        if unique_index in (bin_wise_count_greater_than_50[every_column].index):  ##### if index is present
                            bin_wise_count_greater_than_50[every_column][unique_index] += count_greater_than

                        else:  ##### if index is not present
                            bin_wise_count_greater_than_50[every_column][unique_index] = count_greater_than

    return bin_wise_count_greater_than_50, bin_wise_count_less_than_50, continuous_attributes_bins


def GaussianDistributionTraining(list_of_splits,testing_set_num):
    ''' This function calculate standard deviation and mean by Gaussian distribution to convert continuous variables to discrete bins'''

    gaussian_df = pd.DataFrame()
    standard_deviation_less_than_50 = {}
    mean_less_than_50 = {}
    standard_deviation_greater_than_50 = {}
    mean_greater_than_50 = {}

    for i in range(10):
        if i == testing_set_num:
            continue
        else:
            data = pd.DataFrame(list_of_splits[i])
            gaussian_df = pd.concat([gaussian_df,data])

    columns = ['age','capital-gain','capital-loss','education-num','fnlwgt','hours-per-week']
    df1, df2 = [x for _, x in gaussian_df.groupby(gaussian_df['classification'] == "<=50K")]  #### dividing into 2 dataframes for <=50 and >50

    for every_column in columns:
        standard_deviation_less_than_50[every_column] = df2.loc[:, every_column].std()
        mean_less_than_50[every_column] = df2.loc[:, every_column].mean()

        standard_deviation_greater_than_50[every_column] = df1.loc[:, every_column].std()
        mean_greater_than_50[every_column] = df1.loc[:, every_column].std()

    return standard_deviation_less_than_50,standard_deviation_greater_than_50, mean_less_than_50, mean_greater_than_50


def GaussianDistributionTesting(standard_deviation_less_than_50,
                                        standard_deviation_greater_than_50, mean_less_than_50, mean_greater_than_50, list_of_splits, testing_set_num):
    ''' This function calculates probability of every continuous variable'''


    cont_attr_wise_prob_greater_than_50 = {}
    cont_attr_wise_prob_less_than_50 = {}
    columns = ['age', 'capital-gain', 'capital-loss', 'education-num', 'fnlwgt', 'hours-per-week']

    for every_column in columns:
        # for index, row in list_of_splits[testing_set_num][every_column].iteritems():
        for row in list_of_splits[testing_set_num][every_column]:
            # value = list_of_splits[testing_set_num][row]
            greater_than_value = scipy.stats.norm(mean_greater_than_50[every_column], standard_deviation_greater_than_50[every_column]).pdf(row)
            series = pd.Series([greater_than_value], index=[row])

            if every_column not in cont_attr_wise_prob_greater_than_50:
                cont_attr_wise_prob_greater_than_50[every_column] = series

            else:  #### if column is present in dictionary as key so as not to overwrite
                cont_attr_wise_prob_greater_than_50[every_column].set_value(row, greater_than_value)

            less_than_value = scipy.stats.norm(mean_less_than_50[every_column], standard_deviation_less_than_50[every_column]).pdf(row)
            series = pd.Series([less_than_value], index=[row])

            if every_column not in cont_attr_wise_prob_less_than_50:
                cont_attr_wise_prob_less_than_50[every_column] = series

            else:  #### if column is present in dictionary as key so as not to overwrite
                cont_attr_wise_prob_less_than_50[every_column].set_value(row, less_than_value)

    return cont_attr_wise_prob_greater_than_50, cont_attr_wise_prob_less_than_50

####################################


def ProbabilityOfContinuousAttrTraining(cont_classification_greater_than_50, cont_classification_less_than_50, cont_attr_wise_count_greater_than_50,
            cont_attr_wise_count_less_than_50,continuous_attributes):
    '''This function calculates probability of continuous attributes for training dataset'''

    classification_greater_than_50 = float(cont_classification_greater_than_50)
    classification_less_than_50 = float(cont_classification_less_than_50)
    attr_wise_prob_greater_than_50 = {}
    attr_wise_prob_less_than_50 = {}

    for every_column in continuous_attributes:
        for attr,count in cont_attr_wise_count_greater_than_50[every_column].iteritems():
            value = float(count) / (classification_greater_than_50)
            series = pd.Series([value], index=[attr])

            if every_column not in attr_wise_prob_greater_than_50:
                attr_wise_prob_greater_than_50[every_column] = series

            else:  #### if column is present in dictionary as key so as not to overwrite
                attr_wise_prob_greater_than_50[every_column].set_value(attr, value)

        for attr,count in cont_attr_wise_count_less_than_50[every_column].iteritems():
            value = float(count) / (classification_less_than_50)
            series = pd.Series([value], index=[attr])

            if every_column not in attr_wise_prob_less_than_50:
                attr_wise_prob_less_than_50[every_column] = series

            else:  #### if column is present in dictionary as key
                attr_wise_prob_less_than_50[every_column].set_value(attr, value)

    return attr_wise_prob_greater_than_50,attr_wise_prob_less_than_50


def CountOfDiscreteAttrTraining(list_of_splits,testing_set_num,column_names):
    ''' This function finds the category wise count of all discrete attributes in training dataset'''

    classification_greater_than_50 = 0
    classification_less_than_50 = 0
    attr_wise_count_greater_than_50 = {}
    attr_wise_count_less_than_50 = {}

    for i in range(10):
        if i == testing_set_num:
            continue
        else:
            data = list_of_splits[i]
            values = data['classification'].value_counts()
            classification_less_than_50 += values[0]
            classification_greater_than_50 += values[1]


            for every_column in column_names:

                list_of_unique_value_counts = data[every_column].value_counts()
                for unique_index,everyUnique in list_of_unique_value_counts.iteritems():
                    count_less_than = len(data.loc[(data[every_column] == unique_index) & (data["classification"] == "<=50K"),[every_column,"classification"]])

                    if every_column not in attr_wise_count_less_than_50:    #### if column is not present in dictionary as key
                        series = pd.Series([count_less_than],index = [unique_index])
                        attr_wise_count_less_than_50[every_column] = series

                    else:                               #### if column is present in dictionary as key
                        if unique_index in (attr_wise_count_less_than_50[every_column].index):     ##### if index is present
                            attr_wise_count_less_than_50[every_column][unique_index] += count_less_than

                        else:                                       ##### if index is not present
                            attr_wise_count_less_than_50[every_column][unique_index] = count_less_than

                    ################################# greater thn 50

                    count_greater_than = len(data.loc[(data[every_column] == unique_index) & (data["classification"] == ">50K"), [every_column,"classification"]])

                    if every_column not in attr_wise_count_greater_than_50:    #### if column is not present in dictionary as key
                        series = pd.Series([count_greater_than],index = [unique_index])
                        attr_wise_count_greater_than_50[every_column] = series

                    else:                               #### if column is present in dictionary as key
                        if unique_index in (attr_wise_count_greater_than_50[every_column].index):     ##### if index is present
                            attr_wise_count_greater_than_50[every_column][unique_index] += count_greater_than

                        else:                                       ##### if index is not present
                            attr_wise_count_greater_than_50[every_column][unique_index] = count_greater_than

    return classification_greater_than_50,classification_less_than_50,attr_wise_count_greater_than_50,attr_wise_count_less_than_50


def ProbabilityOfDiscreteAttrTraining(classification_greater_than_50, classification_less_than_50, attr_wise_count_greater_than_50, attr_wise_count_less_than_50,column_names):
    ''' This function calculates the probablity of all categories of discrete attributes in training dataset'''

    classification_greater_than_50 = float(classification_greater_than_50)
    classification_less_than_50 = float(classification_less_than_50)
    attr_wise_prob_greater_than_50 = {}
    attr_wise_prob_less_than_50 = {}

    for every_column in column_names:
        for attr,count in attr_wise_count_greater_than_50[every_column].iteritems():

            value = float(count) / (classification_greater_than_50)
            series = pd.Series([value], index=[attr])

            if every_column not in attr_wise_prob_greater_than_50:
                attr_wise_prob_greater_than_50[every_column] = series

            else:  #### if column is present in dictionary as key so as not to overwrite
                attr_wise_prob_greater_than_50[every_column].set_value(attr, value)

        for attr,count in attr_wise_count_less_than_50[every_column].iteritems():

            value = float(count) / (classification_less_than_50)
            series = pd.Series([value], index=[attr])

            if every_column not in attr_wise_prob_less_than_50:
                attr_wise_prob_less_than_50[every_column] = series

            else:  #### if column is present in dictionary as key
                attr_wise_prob_less_than_50[every_column].set_value(attr, value)

    return attr_wise_prob_greater_than_50,attr_wise_prob_less_than_50



def test_set_predict(data, attr_wise_prob_greater_than_50, attr_wise_prob_less_than_50, column_names,cont_dict, Gaussian):
    '''This function predicts the income for test data (one of the data folds out of 10)'''

    prediction = []
    values = data['classification'].value_counts()
    total_entries = len(data)
    classification_prob_less_than_50 = float(values[0])/total_entries
    classification_prob_greater_than_50 = float(values[1])/total_entries

    for index, row in data.iterrows():
        probability_less_than_50 = 1
        probability_greater_than_50 = 1

        for every_column in column_names :
            test_value = row[every_column]
            if every_column in cont_dict.keys() and Gaussian == 0 :

                temp = test_value/cont_dict[every_column][0]        ### get the appropriate bin
                interval = temp * cont_dict[every_column][0]

                if interval > cont_dict[every_column][1]:       #### if its more than upper bound towards infinity
                    interval = cont_dict[every_column][1]
                if attr_wise_prob_greater_than_50[cont_dict[every_column][2]].get_value(interval) == 0.000 :
                    probability_greater_than_50 = probability_greater_than_50 * 0.000001
                else :
                    probability_greater_than_50 = probability_greater_than_50 * attr_wise_prob_greater_than_50[cont_dict[every_column][2]].get_value(interval)

                if attr_wise_prob_less_than_50[cont_dict[every_column][2]].get_value(interval) == 0.000:
                    probability_less_than_50 = probability_less_than_50 * 0.000001
                else:
                    probability_less_than_50 = probability_less_than_50 * attr_wise_prob_less_than_50[cont_dict[every_column][2]].get_value(interval)

            # else :
            #     probability_greater_than_50 = probability_greater_than_50 * attr_wise_prob_greater_than_50[
            #         cont_dict[every_column][2]].get_value(interval)
            #     probability_less_than_50 = probability_less_than_50 * attr_wise_prob_less_than_50[
            #         cont_dict[every_column][2]].get_value(interval)


            elif every_column not in cont_dict.keys() :
                array = attr_wise_prob_greater_than_50[every_column].index.tolist()
                # if test_value not in [attr_wise_prob_greater_than_50[every_column].index.tolist()]:
                if test_value not in array :
                    probability_greater_than_50 = probability_greater_than_50 * 0.000001
                    probability_less_than_50 = probability_less_than_50 * 0.000001

                else :
                    if attr_wise_prob_greater_than_50[every_column].get_value(test_value) == 0.000:
                        probability_greater_than_50 = probability_greater_than_50 * 0.000001
                    else:
                        probability_greater_than_50 = probability_greater_than_50 * attr_wise_prob_greater_than_50[every_column].get_value(test_value)

                    if attr_wise_prob_less_than_50[every_column].get_value(test_value) == 0.000:
                        probability_less_than_50 = probability_less_than_50 * 0.000001
                    else:
                        probability_less_than_50 = probability_less_than_50 * attr_wise_prob_less_than_50[every_column].get_value(test_value)

        probability_greater_than_50 = classification_prob_greater_than_50 * probability_greater_than_50
        probability_less_than_50 = classification_prob_less_than_50 * probability_less_than_50

        if probability_greater_than_50 > probability_less_than_50:
            prediction.append('>50K')
        else:
            prediction.append('<=50K')
    actual_income = np.array(data['classification'].tolist())
    return prediction, actual_income


def calculate_confusion_matrix(prediction, actual_income):
    '''This function calculates confusion matrix - Accuracy, Precision , Recall and F1'''

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(len(prediction)):
        if actual_income[i] == prediction[i] == "<=50K":
            true_positive += 1
        elif prediction[i] == "<=50K" and actual_income[i] != prediction[i]:
            false_positive += 1
        elif actual_income[i] == prediction[i] == ">50K":
            true_negative += 1
        elif prediction[i] == ">50K" and actual_income[i] != prediction[i]:
            false_negative += 1

    accuracy = float(true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    precision = float(true_positive)/(true_positive + false_positive)
    recall = float(true_positive)/(true_positive + false_negative)
    f1 = 2.0 / ((1 / precision) + (1 / recall))

    print accuracy, precision, recall, f1

    print true_positive, false_positive, true_negative, false_negative
    return accuracy, precision, recall, f1


def main():
    ''' This function iterate loops for all the functions to calculate everything for 9 folds and test on 1 fold, by changing test fold everytime. '''

    # start_time = time.time()

    ################# comment below line to handle missing values by removing
    dataframe, column_names = Handling_Missing_Values.MissingValuesByMode()         ### running code for missing values by mode
    # print ("1", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))

    ################# comment below line to handle missing values by mode
    # dataframe,column_names = Handling_Missing_Values.MissingValuesByRemoving()    ### running code for missing values by removing

    list_of_splits = SplittingDataset(dataframe)

    result_dict = {}

    for testing_set_num in range(10):
        discrete_attributes = ["workclass","education","occupation","marital-status","relationship","race","sex","native-country"]
        classification_greater_than_50, classification_less_than_50, disc_attr_wise_count_greater_than_50, \
        disc_attr_wise_count_less_than_50 = CountOfDiscreteAttrTraining(list_of_splits,testing_set_num, discrete_attributes)

        disc_attr_wise_prob_greater_than_50,disc_attr_wise_prob_less_than_50 = ProbabilityOfDiscreteAttrTraining( classification_greater_than_50,
                            classification_less_than_50, disc_attr_wise_count_greater_than_50,disc_attr_wise_count_less_than_50,discrete_attributes)

        # print ("2", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))


    ##########################comment below 2 lines for implementing Gaussian distribution
        cont_attr_wise_count_greater_than_50, cont_attr_wise_count_less_than_50,continuous_attributes_bins = CountOfContinuousAttrTraining(list_of_splits,testing_set_num)
        Gaussian = 0
        cont_attr_wise_prob_greater_than_50, cont_attr_wise_prob_less_than_50 = ProbabilityOfContinuousAttrTraining(classification_greater_than_50,
                classification_less_than_50, cont_attr_wise_count_greater_than_50,cont_attr_wise_count_less_than_50,continuous_attributes_bins)


    ##########################comment below 2 lines for implementing equal width binning

        # standard_deviation_less_than_50, standard_deviation_greater_than_50, mean_less_than_50, mean_greater_than_50 = GaussianDistributionTraining(list_of_splits, testing_set_num)
        # Gaussian = 1
        #
        # # print ("3except", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))
        # cont_attr_wise_prob_greater_than_50, cont_attr_wise_prob_less_than_50 = GaussianDistributionTesting(standard_deviation_less_than_50,
        #                                 standard_deviation_greater_than_50, mean_less_than_50, mean_greater_than_50, list_of_splits, testing_set_num)

        # print ("3", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))
    ##########################


        disc_attr_wise_prob_greater_than_50.update(cont_attr_wise_prob_greater_than_50)
        disc_attr_wise_prob_less_than_50.update(cont_attr_wise_prob_less_than_50)
        # classes except "classification" attribute
        column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]

        cont_dict = { 'age' : [6, 90, 'age_bins'], 'capital-gain' : [7000, 98000, 'capital_gain_bins'],
                      'capital-loss' : [500, 4000, 'capital_loss_bins'], 'education-num' : [2, 16, 'education_num_bins'],
                      'fnlwgt' : [30000, 780000, 'fnlwgt_bins'],'hours-per-week' : [10, 90, 'hours_per_week_bins']}

        # print ("4", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))

        prediction, actual_income = test_set_predict(list_of_splits[testing_set_num],disc_attr_wise_prob_greater_than_50,disc_attr_wise_prob_less_than_50,column_names,cont_dict, Gaussian)

        accuracy, precision, recall, f1 = calculate_confusion_matrix(prediction, actual_income)
        result_dict[testing_set_num] = [accuracy, precision, recall, f1]

        # print ("5", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))

    print ("\t\t\t\t Accuracy \t\t\t\t Precision \t\t\t\t Recall \t\t\t F1")
    for key, val in result_dict.items():
        print "test fold",key, "=>", val

    # print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # calling main function
    main()