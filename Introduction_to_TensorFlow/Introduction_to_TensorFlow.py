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
