from pandas import DataFrame, Series
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml import sklearn2pmml

iris = load_iris()

X = DataFrame(iris.data, columns = iris.feature_names)
y = Series(iris.target, name = "y").map(lambda x: iris.target_names[x])

def store_csv(df, name):
	df.to_csv("csv/" + name + ".csv", header = True, index = False, sep = ",", na_rep = "N/A")

store_csv(X, "Iris")

def store_pmml(obj, name):
	sklearn2pmml(obj, "pmml/" + name + ".pmml")

def predict(classifier):
	results = DataFrame()
	results[y.name] = classifier.predict(X)
	results[["probability({})".format(cls) for cls in classifier.classes_]] = classifier.predict_proba(X)
	return results

classifier = LogisticRegression(random_state = 42)
classifier.fit(X, y)

store_pmml(classifier, "LogisticRegressionIris")
store_csv(predict(classifier), "LogisticRegressionIris")

classifier = DecisionTreeClassifier(random_state = 42)
classifier.fit(X, y)

store_pmml(classifier, "DecisionTreeIris")
store_csv(predict(classifier), "DecisionTreeIris")
