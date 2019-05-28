from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# import some data
iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=0)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed =  scaler.transform(X_train)

clf = LinearRegression().fit(X_train_transformed, y_train)

predictions = clf.predict(X_test_transformed)

print('Predictions: ', predictions)