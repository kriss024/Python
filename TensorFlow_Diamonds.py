#%%
import warnings 
import math
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from IPython.display import display
print('TensorFlow version: ' + tf.__version__)

#%%
warnings.filterwarnings("ignore")

#%%
diamonds = sns.load_dataset("diamonds")
display(diamonds)

#%%
sns.set()
diamonds_sample = diamonds.sample(5000)
sns.pairplot(diamonds_sample, hue="cut")
# %%
diamonds_cut = diamonds['cut']
le = LabelEncoder()
diamonds_cut_num = le.fit_transform(diamonds_cut)
diamonds['cut_int'] = diamonds_cut_num
display(diamonds)

# %%
uni = diamonds[['cut_int', 'cut']].drop_duplicates()
uni.set_index('cut_int', inplace=True)
diamonds_dict = dict(zip(uni.index, uni.iloc[:, 0]))
print(diamonds_dict)

# %%
diamonds.info()

# %%
diamonds_depends = diamonds.drop(['cut', 'cut_int'], axis=1)
display(diamonds_depends)

# %%
catCols = diamonds_depends.select_dtypes("object").columns.tolist()
print(catCols)

# %%
# Convert labels to categorical one-hot encoding
diamonds_dummies = pd.get_dummies(diamonds_depends, columns = catCols)
display(diamonds_dummies)

# %%
# calculate the correlation matrix
corr = diamonds_dummies.corr()
# plot the heatmap
sns.heatmap(corr, 
xticklabels=corr.columns,
yticklabels=corr.columns)

# %%
target = pd.get_dummies(diamonds['cut'], prefix = 'cut')
target_name = target.columns.tolist()
print(target_name)
display(target)
n_outout = len(target_name)
print(n_outout)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diamonds_dummies, target, test_size=0.33, random_state=42)
display(X_train)
display(y_train)

# %%
from sklearn.preprocessing import Normalizer
norm = Normalizer()
names = X_train.columns
X_train_norm = norm.fit_transform(X_train)
X_test_norm = norm.transform(X_test)
X_train_pd = pd.DataFrame(X_train_norm)
X_test_pd = pd.DataFrame(X_test_norm)
X_train_pd.columns = names
X_test_pd.columns = names
display(X_train)
n_input = len(X_train_pd.columns)
print(n_input)

# %%
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation

model = Sequential()
model.add(Flatten(input_dim=n_input))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(n_outout, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
# %%
total_rows = X_train_pd.shape[0]
batch_size=int(total_rows/10)
print('Total rows:'+str(total_rows)+',  Batch size:'+str(batch_size))
# %%
model.fit(X_train_pd, y_train, epochs=100, verbose=1, batch_size=batch_size, use_multiprocessing=True)
# %%
# evaluate the model
scores = model.evaluate(X_test_pd, y_test, batch_size=batch_size, use_multiprocessing=True)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# %%
# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")
del model  # deletes the existing model

# %%
# load and evaluate a saved model
from keras.models import load_model
# load model
model = load_model('model.h5')
# summarize model.
model.summary()
print("Loaded model from disk")

# %%
# evaluate the model
scores = model.evaluate(X_test_pd, y_test, batch_size=batch_size, use_multiprocessing=True)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# %%
y_proba = model.predict(X_test)
y_test['Prediction'] = np.argmax(y_proba, axis=-1)
y_test['Prediction'].replace(diamonds_dict, inplace=True)
y_test_ds = diamonds.join(y_test, how= 'inner')
pd.crosstab(y_test_ds['cut'], y_test_ds['Prediction'])
