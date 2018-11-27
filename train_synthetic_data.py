from utils import data_generator
from visualize import *
import numpy as np


num_sample = 20
mean0 = [0, 0]
mean1 = [3, 3]
cov0 = [[.1, 0], [0, .1]]
cov1 = [[.1, 0], [0, .1]]
means = np.array([mean0, mean1])
cov = np.array([cov0, cov1])

features, adj, labels = data_generator(means=means, covariances=cov, num_sample=num_sample, threshold=1)
colors = ['b' if label == 0 else 'r' for label in labels]
plt.scatter(features[:, 0], features[:, 1], s=10, c=colors)
plt.show()
affinity_visualize(adj, features, labels, num_sample, 2)
