# Pool Water Quality Prediction using K-nearest Neighbor

This project uses the K-nearest Neighbor classifier to predict the quality of pool water based on attributes such as pH level, chlorine content, water temperature, turbidity, and then classifies them as good or bad quality.

1. In a new jupyter notebook file I ran the command to install the required packages

!pip install scikit-learn

2. Import the nescessary libraries

import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

3. Loaded the dataset onto the program and outputted

df=pd.read_csv("WaterQual.csv")
df.head()

4. Preprocessed the dataset, did Train/Test Split, and Feature Scaled the model

X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=69)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

5. Made the prediction, ("I used the number 3 because it was the most accurate, the second best number was 5.")

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

6. Evaluates the Algorithm

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

7. I also did use Gaussian NB to test out the different accuracy as well

model = GaussianNB()
model.fit(X_train, y_train)

predictions=model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))

8. Should generate the output with confusion matrix and precision, recall, f1-score, support

References: https://www.kdnuggets.com/2022/07/knearest-neighbors-scikitlearn.html
I used this blog as a reference for the base of my code.
