import pandas as pd
import numpy as np
import random

class NB:
    def open(self): 
        df = pd.read_csv("~/Desktop/DS monaksha/monaksha/Iris.csv")

        feture_vector = []

        for i in range(len(df)):
            feture = []
            feture.append(df["SepalLengthCm"][i])
            feture.append(df["SepalWidthCm"][i])
            feture.append(df["PetalLengthCm"][i])
            feture.append(df["PetalWidthCm"][i])
            feture.append(df["Species"][i])
            feture_vector.append(feture)

        random.shuffle(feture_vector)

        # feture_vector = feture_vector[:10]

        label = []
        for i in range(len(feture_vector)):
            label.append(feture_vector[i][4])
            feture_vector[i] = feture_vector[i][:4]
        
        return np.array(feture_vector),np.array(label)

    def fit(self,X,y):
        sample_num , feature_num = X.shape #(3 , 10)
        self.classes = np.unique(y)
        class_num = len(self.classes)

        self.mean = np.zeros((class_num,feature_num),dtype=np.float64)
        self.var = np.zeros((class_num,feature_num),dtype=np.float64)
        self.priors = np.zeros((class_num),dtype=np.float64)

        for index , label in enumerate(self.classes):
            X_label = X[label == y]
            self.mean[index, : ] = X_label.mean(axis=0) #?
            self.var[index, : ] = X_label.var(axis=0) #?
            self.priors[index] = X_label.shape[0] / sample_num #p(y) = (class 1) /p(all classes)
    
    def predict(self,X):
        y_pred = []
        for x in X:
            posteriors = []
            for index , label in enumerate(self.classes):
                prior = np.log(self.priors[index])
                posterior = np.sum(np.log(self._gaussian(index,x)))
                posterior += prior
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return y_pred

    def _gaussian(self,index,x):
        mean = self.mean[index]
        var = self.var[index]
        Gaussian = np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
        return Gaussian
    
    def accuracy(self,labels, y_pred):
        return np.sum(labels == y_pred) / len(labels)

nb = NB()
x_train , y_train = nb.open()
x_test , y_test = nb.open()

x_train = x_train[:70]
y_train = y_train[:70]
x_test = x_test[:30]
y_test = y_test[:30]

nb.fit(x_train,y_train)

predictions = nb.predict(x_test)

print("accuracy = ",nb.accuracy(y_test, predictions))