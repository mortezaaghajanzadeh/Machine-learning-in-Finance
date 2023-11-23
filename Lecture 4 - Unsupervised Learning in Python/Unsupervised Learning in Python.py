#%% Unsupervised Learning in Python Datacamp
#%% Clustering for dataset exploration
#%% import common libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import datasets
plt.style.use('ggplot')
#%% import libraries coursework
from sklearn.cluster import KMeans

#%%
# Iris dataset in scikit-learn
iris = datasets.load_iris()
print(iris.keys())
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())
#%%
samples = iris.data
model = KMeans(n_clusters=3)
model.fit(samples)
labels = model.predict(samples)
print(labels)
# %%
new_samples = iris.data[:5,:]
new_labels = model.predict(new_samples)
print(new_labels)
