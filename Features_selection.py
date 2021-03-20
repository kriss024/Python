# %%
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from IPython.display import display
# %%
n = 1000
x1 = np.linspace(0.0, 100.0, n)
x2 = np.random.uniform(0.0, 100.0, n)
bins = pd.cut(x1, bins=10, labels=False) + 1
y = bins % 2
dataset = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
display(dataset)
# %%
dataset.info()
# %%
columns = dataset.columns.tolist()
target = ['y']
feature_names = [item for item in columns if item not in target]
rows = len(dataset)
print('All columns: ' + str(columns))
print('Feature names: ' + str(feature_names))
print('Target: ' + str(target))
print('Number of rows: ' + str(rows))
# %%
pg.anova(data=dataset, dv='x1', between='y', detailed=True)
# %%
pg.anova(data=dataset, dv='x2', between='y', detailed=True)
# %%
X = dataset[['x1']]
y = dataset[target]
rows = len(X)
min_samples_leaf = int(rows*0.05)
classifier = DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf)
classifier.fit(X, y)
# %%
X_columns = X.columns.tolist()
text_representation = export_text(classifier, feature_names=X_columns)
print(text_representation)
# %%
data = export_graphviz(classifier , out_file=None, feature_names=X_columns)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('decisiontree.png')
# %%
img = pltimg.imread('decisiontree.png')
plt.imshow(img)
plt.axis('off')
plt.show()
# %%
y_pred = classifier.predict(X)
print(confusion_matrix(y, y_pred))
accuracy = accuracy_score(y, y_pred)
print('Accuracy: '+'{:.2%}'.format(accuracy))
# %%
def DecisionTreeAccuracy(X, y, criterion, p):
    rows = len(X)
    min_samples_leaf = int(rows*p)
    classifier = DecisionTreeClassifier(criterion=criterion, min_samples_leaf=min_samples_leaf)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    return accuracy_score(y, y_pred)
# %%
print("Feature ranking:")
for elem in feature_names:
    X = dataset[[elem]]
    y = dataset[target]
    accuracy = DecisionTreeAccuracy(X ,y, 'gini', 0.05)
    print(elem+ ', accuracy: '+'{:.2%}'.format(accuracy))