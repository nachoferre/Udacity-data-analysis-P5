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
file_names = ['salary', 'total\npayments', 'bonus', 'total\nstock_value', 'expenses',
                              'to_messages', 'from_poi_to\nthis_person', 'from\nmessages',
                              'from_this\nperson_to_poi', 'shared_receipt\nwith_poi', 'total_income']
ind = np.arange(len(file_names))
best_scores = [0.0, 0.028768699654775496, 0.0020833333333333346, 0.042361111111111245, 0.36127945165089814, 0.0,
               0.045045045045045085, 0.0, 0.059615666220563475, 0.11952932771481159, 0.34131736526946166]
# for name in file_names:
#     with open(name + "_best_result.txt", "r") as results:
#         aux = results.readline()
#         aux = aux.split(": ")[1]
#         aux = float(aux.strip())
#         best_scores.append(aux)

fig, ax = plt.subplots()
rect = ax.barh(ind, best_scores, width, color='g')
ax.set_yticks(ind + width / 2)
ax.set_yticklabels(file_names)
ax.set_ylabel('Scores')
ax.set_title('Feature Importances')

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



