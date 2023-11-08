#%%
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# %%
# Simulate recovery rate data.
np.random.seed(58)
n = 500
# Set true values pf parameters.
a, b, c = 0.0, 0.2, 1.0
data = np.random.triangular(a, b, c, n)
# Plot histogram.
plt.hist(data, bins=30, density=True, alpha=0.5)
plt.title('Simulated Recovery Rates')
plt.xlabel('Rate')
plt.ylabel('Density')
plt.show()
# %%
input_tensor = Input(shape=(1,),name = 'Input')
output_layer = Dense(3, name='Output')
output_tensor = output_layer(input_tensor)
model = Model(inputs=input_tensor, outputs=output_tensor)

