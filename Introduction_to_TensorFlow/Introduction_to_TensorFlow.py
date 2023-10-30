#%%
from tensorflow import *

#%%
a0 = Variable([1,2,3,4,5,6,7,8,9,10], dtype=float32)
a1 = Variable([1,2,3,4,5,6,7,8,9,10], dtype=int16)
# Define a constant
b = constant(2, dtype=float32)

c0 = a0 * b
c0
#%%
A = ones([2,3,4])
reduce_sum(A, axis=0), reduce_sum(A, axis=1), reduce_sum(A, axis=2)
#%%
# element-wise products
wealth = constant([[11,50],
             [7,2],
             [4,60],
             [3,0],
             [25,10]])
reduce_sum(wealth, axis=0), reduce_sum(wealth, axis=1)
#%%
x = Variable(-1.0)
with GradientTape() as tape:
    tape.watch(x)
    y = multiply(x, x)
# Evaluates the gradient of y at x = -1
dy_dx = tape.gradient(y, x)
dy_dx.numpy()
#%%
gray = random.uniform([2,2,3],maxval=255, dtype='int32')
print(gray.numpy())
gray = reshape(gray, [2*2,3])
gray.numpy()
#%% Import data
features = constant([1,2,3,4,5,6,7,8,9,10], dtype=float32)
targets = constant([2,4,6,8,10,12,14,16,18,20], dtype=float32)
slope = Variable(0.5)

def linear_regression(intercept,slope = slope, features = features):
    return intercept + features*slope
def loss_function(intercept,slope,targets = targets, features = features):
    predictions = linear_regression(intercept,slope)
    return keras.losses.mse(targets,predictions)

loss_function(0.1,0.1)
#%%
inputs = constant([[1,35]], dtype=float32)
weights = Variable([[-0.05],[-0.01]])
bias = Variable([0.5])
product = matmul(inputs, weights)
dense = keras.activations.sigmoid(product + bias)
#%%
# inputs = constant(data, dtype=float32)
# dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)
# dense2 = keras.layers.Dense(5, activation='sigmoid')(dense1)
# outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)
#%%
import numpy as np
young, old = 0.3,0.6
low_bill, high_bill = 0.1,0.5

young_high = 1.0 * young + 2.0 * high_bill
young_low = 1.0 * young + 2.0 * low_bill
old_high = 1.0 * old + 2.0 * high_bill
old_low = 1.0 * old + 2.0 * low_bill

print(young_high - young_low)
print(old_high - old_low)

print(keras.activations.sigmoid(young_high).numpy() - keras.activations.sigmoid(young_low).numpy())
print(keras.activations.sigmoid(old_high).numpy() - keras.activations.sigmoid(old_low).numpy())
#%%
bill_amounts = np.random.random((3000, 3))
inputs = constant(bill_amounts,dtype=float32)
dense1 = keras.layers.Dense(3, activation='relu')(inputs)
dense2 = keras.layers.Dense(2, activation='relu')(dense1)
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)
outputs
#%%
borrower_features = np.random.random((3000, 10))
inputs = constant(borrower_features,dtype=float32)

# Define first dense layer
dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(8, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(6, activation='softmax')(dense2)
outputs.numpy()
# %%
def model(bias,weights,features = borrower_features):
    product = matmul(features,weights)
    return keras.activations.sigmoid(product + bias)
def loss_function(bias,weights,features = borrower_features):
    predictions = model(bias,weights)
    return keras.losses.binary_focal_crossentropy(targets,predictions)
opt = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.9)
opt.minimize(lambda: loss_function(bias,weights), var_list=[bias,weights])
# %%
weights = Variable(random.normal([500,500]))
weights = Variable(random.truncated_normal([500,500]))

dense = keras.layers.Dense(32, activation='relu')
dense = keras.layers.Dense(32, activation='relu', kernel_initializer='zeros')

inputs = np.array(borrower_features, dtype=np.float32)
dense1 = keras.layers.Dense(32, activation='relu')(inputs)
dense2 = keras.layers.Dense(16, activation='relu')(dense1)
dropout = keras.layers.Dropout(0.25)(dense2)
outputs = keras.layers.Dense(1, activation='sigmoid')(dropout)
outputs.numpy()
#%%
# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))


# Initialize the layer 1 bias
b1 = Variable(ones([7]))

# Define the layer 2 weights
w2 = Variable(random.normal([7,1]))

# Define the layer 2 bias
b2 = Variable([0])
# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
	# Apply relu activation functions to layer 1
	layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout rate of 0.25
	dropout = keras.layers.Dropout(0.25)(layer1)
	return keras.activations.sigmoid(matmul(dropout, w2) + b2)

# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
	predictions = model(w1, b1, w2, b2)
	# Pass targets and predictions to the cross entropy loss
	return keras.losses.binary_crossentropy(targets, predictions)
# Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), 
                 var_list=[w1, b1, w2, b2])

# Make predictions with model using test features
model_predictions = model(w1, b1, w2, b2, test_features)

# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)