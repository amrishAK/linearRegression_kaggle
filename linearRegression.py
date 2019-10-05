import datetime

import pandas as pd
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer
 
# Dataset Path
testDS_path = "tcd ml 2019-20 income prediction test (without labels).csv"
trainDS_path = "tcd ml 2019-20 income prediction training (with labels).csv"

def ManagingNulls(dataFrame):
    
    #Year of Record [dataType = float64] -> current year
    currentYear = float(datetime.datetime.now().year)
    dataFrame['Year of Record'] = dataFrame['Year of Record'].fillna(currentYear)

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
    dataFrame['University Degree'] = dataFrame['University Degree'].replace(['No','0'],'No Degree')

    #Hair Color => ['Unknown','0'] -> Unknown Hair Color
    dataFrame['Hair Color'] = dataFrame['Hair Color'].replace(['Unknown','0'],'Unknown Hair Color')

    return dataFrame

def FeatureExtraction(dataFrame):

    #create a label binary format
    lbFormat = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    
    #Extract Genders and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(lbFormat.fit_transform(dataFrame['Gender']),columns=lbFormat.classes_,index=dataFrame.index))

    #Remove the Gender Column
    dataFrame = dataFrame.drop(['Gender'], axis = 1)

    #Extract University Degree and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(lbFormat.fit_transform(dataFrame['University Degree']),columns=lbFormat.classes_,index=dataFrame.index))

    #Remove the University Degree Column
    dataFrame = dataFrame.drop(['University Degree'], axis = 1)

    #Extract Hair Color and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(lbFormat.fit_transform(dataFrame['Hair Color']),columns=lbFormat.classes_,index=dataFrame.index))

    #Remove the Hair Color Column
    dataFrame = dataFrame.drop(['Hair Color'], axis = 1)

    return dataFrame

def ModelCreation(xFrame, yFrame):
    return LinearRegression().fit(xFrame,yFrame)


def Preprocessing(dataFrame):
    
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
    #linearModel = ModelCreation(processedTrainingFrame,trainingFrame['Income'])

    testFrame = pd.read_csv(testDS_path)
    processedTestFrame = Preprocessing(testFrame)

    print(processedTrainingFrame)
    print(processedTestFrame)

    #prediction
    #linearModel.predict(processedTestFrame)


if __name__ == '__main__':
    run()