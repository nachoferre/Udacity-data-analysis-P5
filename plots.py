import matplotlib.pyplot as plt
import numpy as np
import json


def autolabel( rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + 0.01,
                round(height, 2),
                ha='center')
# the x locations for the groups
width = 0.35
file_names = ["gaussian_estimator", "neural_estimator", "tree_estimator", "logistic_estimator"]
ind = np.arange(len(file_names))
best_scores = []
for name in file_names:
    with open(name + "_best_result.txt", "r") as results:
        aux = results.readline()
        aux = aux.split(": ")[1]
        aux = float(aux.strip())
        best_scores.append(aux)

fig, ax = plt.subplots()
rect = ax.bar(ind, best_scores, width, color='g')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(file_names)
autolabel(rect, ax)
plt.ylim([0, 1])
ax.set_ylabel('Scores')
ax.set_title('Algorithms calculated')

plt.show()

json_data = open("neural_estimator_results.json").read()
data = json.loads(json_data)
ind = np.arange(len(data["mean_test_score"][:92]))
print ind
fig, ax = plt.subplots()
width = 0.35
ax.bar(ind, data["mean_test_score"][:92], width)
ax.set_ylabel('Scores')
ax.set_title('Neural Network')
plt.show()



