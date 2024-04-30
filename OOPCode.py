import pandas as pd
import numpy as np
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

class dataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def loadData(self):
        self.data = pd.read_csv(self.file_path)
    
    def dropNA(self):
        self.data.dropna(inplace=True)
    
    def InputOutput(self, targetCol):
        self.output_df = self.data[targetCol]
        self.input_df = self.data.drop([targetCol], axis=1)

class modelHandler:
    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.x_train, self.x_test, self.y_train, self.y_test, self,y_pred = [None] * 6

    def dropCol(self, column):
        self.input = self.input.drop(column, axis=1)

    def createMean(self, col):
        return statistics.mean(self.x_train[col])
    
    def createMode(self, col):
        return statistics.mode(self.x_train[col])

    def splitData(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.input, self.output, test_size=test_size, random_state=random_state)

    def OneHot(self, columns):
        encoder = OneHotEncoder()
        xtrainsub = self.x_train[columns]
        trainEnc = pd.DataFrame(encoder.fit_transform(xtrainsub).toarray(), columns=encoder.get_feature_names_out())
        self.x_train = self.x_train.reset_index(drop=True)
        self.x_train = pd.concat([self.x_train, trainEnc], axis=1)

        xtestsub = self.x_test[columns]
        testData = pd.DataFrame(encoder.transform(xtestsub).toarray(), columns=encoder.get_feature_names_out())
        self.x_test = self.x_test.reset_index(drop=True)
        self.x_test = pd.concat([self.x_test, testData], axis=1)

        self.x_train.drop(columns, axis=1, inplace=True)
        self.x_test.drop(columns, axis=1, inplace=True)

        self.exportEnc(encoder, path='OneHotEncoder.pkl')
    
    def exportEnc(self, encoder, path='OneHotEncoder.pkl'):
        with open(path, "wb") as encoder_file:
            pickle.dump(encoder, encoder_file)
    
    def createModel(self):
        self.model = XGBClassifier(
            n_estimator=150,
            max_depth=7,
            learning_rate=0.1,
            gamma=0.4,
            colsample_bytree=0.8
        )

    def train(self):
        self.model.fit(self.x_train, self.y_train)
    
    def predict(self):
        self.y_pred = self.model.predict(self.x_test)

    def classReport(self):
        print("\nClassification Report\n")
        print(classification_report(self.y_test, self.y_pred))
    
    def modelEval(self):
        pred = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, pred)
    
    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
    
file_path = 'data_B.csv'
data_handler = dataHandler(file_path)
data_handler.loadData()
data_handler.InputOutput('churn')
data_handler.dropNA()

input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = modelHandler(input_df, output_df)
model_handler.dropCol(['Unnamed: 0', 'id', 'CustomerId', 'Surname'])

model_handler.splitData()
model_handler.OneHot(['Geography', 'Gender'])

model_handler.createModel()
model_handler.train()
model_handler.predict()

print("Model Accuracy: ", model_handler.modelEval())
model_handler.classReport()

model_handler.save_model_to_file('finalized_model.pkl')