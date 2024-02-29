from data_preparation import ZZFeatureMap
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



iris_data = load_iris()

features = iris_data.data
num_features = features.shape[1]
print("num_features: ", num_features)
labels = iris_data.target
features = MinMaxScaler().fit_transform(features)

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1, insert_barriers=True, entanglement='linear')
print(feature_map)
