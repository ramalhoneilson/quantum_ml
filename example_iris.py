from qiskit.primitives import Sampler
from sklearn.model_selection import train_test_split

from n_local import RealAmplitudes
from data_preparation import ZZFeatureMap
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC

random_seed = 123

iris_data = load_iris()

features = iris_data.data
num_features = features.shape[1]
labels = iris_data.target
#classical preprocessing
features = MinMaxScaler().fit_transform(features)

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)

print(feature_map.parameters)
feature_map.decompose().draw(output="mpl")
plt.show()

#
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", fold=20)
plt.show()



optimizer = COBYLA(maxiter=100)

sampler = Sampler()

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)
#
def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

vqc:VQC = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)



train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=random_seed
)

vqc.fit(train_features, train_labels)

# save the model
vqc.save("vqc_iris")

# train_score_q4 = vqc.score(train_features, train_labels)
#
# test_score_q4 = vqc.score(test_features, test_labels)
#
# print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
# print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")
#
