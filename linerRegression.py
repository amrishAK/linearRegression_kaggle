import datetime

import pandas as pd
import numpy
from sklearn.linear_model import LinearRegression
 
# Dataset Path
testDS_path = "tcd ml 2019-20 income prediction test (without labels).csv"
trainDS_path = "tcd ml 2019-20 income prediction training (with labels).csv"

def ManagingNulls(dataFrame):
    
    #Year of Record [dataType = float64] -> current year
    currentYear = float(datetime.datetime.now().year)
    dataFrame['Year of Record'] = dataFrame['Year of Record'].fillna(currentYear)

    #Gender [dataType = object] -> Other Gender
    dataFrame['Gender'] = dataFrame['Gender'].fillna('Other Gender')

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
    return dataFrame

def FeatureExtraction(dataFrame):
    return dataFrame

def ModelCreation(xFrame, yFrame):
    return LinearRegression().fit(xFrame,yFrame)


def Preprocessing(dataFrame):
    
    #drop income form  data frame
    dataFrame = dataFrame.drop(['Income'],axis = 1)
   
    #Managing nulls
    dataFrame = ManagingNulls(dataFrame)

    #formatting columns
    dataFrame = FormattingColumn(dataFrame)

    #feature extraction
    dataFrame = FeatureExtraction(dataFrame)


    return dataFrame

def run():

    #load data
    trainingFrame = pd.read_csv(trainDS_path)

    #preprocessing
    processedTrainingFrame = Preprocessing(trainingFrame) 

    #create model and train
    linearModel = ModelCreation(processedTrainingFrame,trainingFrame['Income'])

    testFrame = pd.read_csv(testDS_path)
    processedTestFrame = Preprocessing(testFrame)

    #prediction
    linearModel.predict(processedTestFrame)

    print(processedTestFrame)
    print(processedTrainingFrame)

if __name__ == '__main__':
    run()