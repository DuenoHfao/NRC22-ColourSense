import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("colour.data")

convert = preprocessing.LabelEncoder()
distance = convert.fit_transform(list(data["distance"]))
red = convert.fit_transform(list(data["R"]))
green = convert.fit_transform(list(data["G"]))
blue = convert.fit_transform(list(data["B"]))
white = convert.fit_transform(list(data["W"]))
colour = convert.fit_transform(list(data["C"]))

predict = "colour"
x = list(zip(distance, red, green, blue, white))
# x is the input variables
y = list(colour)
# y is the prediction

with open("optimal.txt", "r") as findAcc:
    optValues = findAcc.readlines()
    if len(optValues) == 0:
        optimalI = 0
        highestAcc = 0
    else:
        highestAcc = float(optValues[1])
        optimalI = int(optValues[0])

for i in range(1, 45):
    for v in range(20):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        if accuracy > highestAcc:
            highestAcc = accuracy
            optimalI = i

with open("optimal.txt", "w") as opt:
    opt.write(str(optimalI))
    opt.write("\n")
    opt.write(str(highestAcc))

predicted = model.predict(x_test)
names = ["black", "green", "red", "yellow"]

for x in range(len(predicted)):
    if predicted[x] != y_test[x]:
        print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
        # n = model.kneighbors([x_test[x]], optimalI, True)
        # print("Neighbours: ", n)
        # print("\n")
print(accuracy)
print("\n")

userInput = input("Input RGBW values(format: ww,xx,yy,zz): ")
userData = []
x = userInput.split(",")
for data in x:
    userData.append(int(data))