import numpy as np
import os
from os.path import join
import argparse
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import json
from datetime import datetime


def get_dico():
    classes = []
    PATH = os.path.dirname(__file__)
    directory = join(PATH,'./data/wikipaintings_train')
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    class_indices = dict(zip(classes, range(len(classes))))
    return class_indices


y_true = np.load("./y_true.npy")
print(y_true)
print(len(y_true))
y_pred = np.load("./y_pred.npy")
y_pred2 = np.array([x[0] for x in y_pred])
#print(y_pred2)
#print(len(y_pred2))
conf_arr=confusion_matrix(y_true, y_pred2)
np.savetxt("confusion.csv", conf_arr, delimiter=",", fmt="%3.0d")

dico = get_dico()

new_conf_arr = []
for row in conf_arr:
	new_conf_arr.append(row / sum(row))

plt.matshow(new_conf_arr)
plt.yticks(range(25), dico.keys())
plt.xticks(range(25), dico.keys(), rotation=90)
plt.colorbar()
#plt.show()
plt.savefig("confusion", bbox_inches="tight", pad_inches=0)