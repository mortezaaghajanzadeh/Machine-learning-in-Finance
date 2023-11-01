#%%
import pandas as pd
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
#%% Input
input_tensor = Input(shape=(1,))
print(input_tensor)
#%% Ouput
output_layer = Dense(1,name = 'Predicted-Score-Diff')
output_tensor = output_layer(input_tensor)
#%% Model
model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mae')
print(model.summary())
plot_model(model, to_file='model.png')
# %%
import matplotlib.pyplot as plt
img = plt.imread('model.png')
plt.imshow(img)
plt.show()
