import datetime

import pandas as pd
import numpy
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset Path
testDS_path = "tcd ml 2019-20 income prediction test (without labels).csv"
trainDS_path = "tcd ml 2019-20 income prediction training (with labels).csv"

#Store uniques
professionUniques = None
countryUniques = None

def ManagingNulls(dataFrame):
    
    #Year of Record [dataType = float64] -> mean
    #currentYear = float(datetime.datetime.now().year)
    dataFrame['Year of Record'] = dataFrame['Year of Record'].fillna(dataFrame['Year of Record'].median())

    #Gender [dataType = object] -> Unknown Gender
    dataFrame['Gender'] = dataFrame['Gender'].fillna('Unknown Gender')

    #Age [dataType = float64] -> mean
    dataFrame['Age'] = dataFrame['Age'].fillna(dataFrame['Age'].mean())

    #Profession [dataType = object] -> No Profession
    dataFrame['Profession'] = dataFrame['Profession'].fillna('No Profession')

    #University Degree [dataType = object] -> No Degree
    dataFrame['University Degree'] = dataFrame['University Degree'].fillna('No Degree')

    #Hair Color [dataType = object] -> No Hair 
    dataFrame['Hair Color'] = dataFrame['Hair Color'].fillna('No Hair')

    return dataFrame

def FormattingColumn(dataFrame):
    
    #Gender => ['0','unknown'] -> Unknown Gender | ['other'] -> Other Gender
    dataFrame['Gender'] = dataFrame['Gender'].replace(['0','unknown'],'Unknown Gender')
    dataFrame['Gender'] = dataFrame['Gender'].replace(['other'],'Other Gender')
    
    #University Degree => ['No','0'] -> No Degree 
    dataFrame['University Degree'] = dataFrame['University Degree'].replace(['No'],'No Degree')
    dataFrame['University Degree'] = dataFrame['University Degree'].replace(['0'],'0 Degree')

    #Hair Color => ['Unknown','0'] -> Unknown Hair Color
    dataFrame['Hair Color'] = dataFrame['Hair Color'].replace(['Unknown','0'],'Unknown Hair Color')

    return dataFrame

def FeatureExtraction(dataFrame,columnName):

    # OneHeartEncoder
    encoder = OneHotEncoder(sparse = False, handle_unknown = 'ignore')

    #reshape the column
    column = dataFrame[columnName]
    column = numpy.array(column).reshape(-1,1)

    #Extract and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(encoder.fit_transform(column),columns=encoder.categories_,index=dataFrame.index))

    #Remove the Column
    dataFrame = dataFrame.drop([columnName], axis = 1)

    return dataFrame

def ExtractingSplFeatures(uniques,dataFrame,columnName):
    
    # OneHeartEncoder
    encoder = OneHotEncoder(categories = [uniques],sparse = False, handle_unknown = 'ignore')

    #reshape the column
    column = dataFrame[columnName]
    column = numpy.array(column).reshape(-1,1)

    #Extract the column and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(encoder.fit_transform(column),columns=encoder.categories_,index=dataFrame.index))

    #Remove the profession Column
    dataFrame = dataFrame.drop([columnName], axis = 1)

    return dataFrame

def ScalingColumns(dataFrame,columnName):
    scaler = preprocessing.MinMaxScaler()
    column = dataFrame[columnName].astype(float)
    column = numpy.array(column).reshape(-1,1)
    scaledColumn = scaler.fit_transform(column)
    normalizeColumn = pd.DataFrame(scaledColumn)
    dataFrame[columnName] = normalizeColumn
    return dataFrame

def Preprocessing(dataFrame):
    
    #Managing nulls
    dataFrame = ManagingNulls(dataFrame)

    #formatting columns
    dataFrame = FormattingColumn(dataFrame)

    #feature extraction gender
    dataFrame = FeatureExtraction(dataFrame,'Gender')

    #feature extraction uni degree
    dataFrame = FeatureExtraction(dataFrame,'University Degree')

    # #scaling age
    # dataFrame = ScalingColumns(dataFrame,'Age')

    # #scaling height
    # dataFrame = ScalingColumns(dataFrame,'Body Height [cm]')

    # #scaling Year of Record
    # dataFrame = ScalingColumns(dataFrame,'Year of Record')

    #scaling Year of Record
    #dataFrame = ScalingColumns(dataFrame,'Size of City')

    #Initial attempt drop -> Instance, Wear Glasses and city size
    dataFrame = dataFrame.drop(['Size of City'], axis = 1)
    dataFrame = dataFrame.drop(['Hair Color'], axis = 1)
    # dataFrame = dataFrame.drop(['Wears Glasses'], axis = 1)
    dataFrame = dataFrame.drop(['Instance'], axis = 1)

    return dataFrame

def PreprocessingTrainingDS():
    #load data
    trainingFrame = pd.read_csv(trainDS_path)

    #preprocessing - basic
    processedTrainingFrame = Preprocessing(trainingFrame)

    #store uniques
    global professionUniques
    professionUniques = processedTrainingFrame['Profession'].unique()
    global countryUniques
    countryUniques = processedTrainingFrame['Country'].unique()

    #Extract profession
    processedTrainingFrame = ExtractingSplFeatures(professionUniques,processedTrainingFrame,'Profession')

    #Extract Country
    processedTrainingFrame = ExtractingSplFeatures(countryUniques,processedTrainingFrame,'Country')

    # remove the negtive income rows
    processedTrainingFrame = processedTrainingFrame[processedTrainingFrame['Income in EUR'] > 0]

    # remove outliers
    processedTrainingFrame = processedTrainingFrame[processedTrainingFrame['Income in EUR'] < 2600000]

    return processedTrainingFrame.drop(['Income in EUR'],axis = 1), processedTrainingFrame['Income in EUR']


def PreprocessingTestDS():
    testFrame = pd.read_csv(testDS_path)

    instance = testFrame['Instance']

    #Remove income Column
    testFrame = testFrame.drop(['Income'],axis = 1)

    processedTestFrame = Preprocessing(testFrame)

    #load the training data
    trainingFrame = pd.read_csv(trainDS_path)

    #Extract profession
    processedTestFrame = ExtractingSplFeatures(professionUniques,processedTestFrame,'Profession')
    
    #Extract Country
    processedTestFrame = ExtractingSplFeatures(countryUniques,processedTestFrame,'Country')
    
    return processedTestFrame,instance

def ModelCreation(xFrame, yFrame):
    return BayesianRidge().fit(xFrame,yFrame)

def run():

    print("stated training data preprocessing")
    #load and preprocess training data
    (xDataFrame,yDataFrame) = PreprocessingTrainingDS()

    #take log for yDataFrame
    yDataFrame = numpy.log(yDataFrame)

    #split the validation data
    xDataFrame, xDataFrameValidate, yDataFrame, yDataFrameValidate = train_test_split(xDataFrame, yDataFrame, test_size = 0.2, random_state = 0)
   
    print("stated creating linear model")
    #create model and train
    regressionModel = ModelCreation(xDataFrame,yDataFrame)

    print("stated testing data preprocessing")
    #load and preprocess test data
    (testDataFrame,testInstance) = PreprocessingTestDS()

    print("stated prediction")   

    #prediction
    predictionValidate = regressionModel.predict(xDataFrameValidate)
    prediction = regressionModel.predict(testDataFrame)
    yDataFrameValidate = numpy.exp(yDataFrameValidate)

    #take exponent to bring values back to normal
    predictionValidate = numpy.exp(predictionValidate)
    prediction = numpy.exp(prediction)

    #pushing to file
    outputDataFrame = pd.DataFrame(testInstance)
    outputDataFrame['Income'] = pd.DataFrame(prediction)
    print("pusing to a file") 
    outputDataFrame.to_csv('tcd ml 2019-20 income prediction submission file.csv')
    print("Mean Square Error is " + str(numpy.sqrt(mean_squared_error(yDataFrameValidate, predictionValidate))))

if __name__ == '__main__':
    run()