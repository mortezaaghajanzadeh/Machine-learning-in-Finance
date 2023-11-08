#%%
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from keras.optimizers import Adam

#%% Input
input_tensor = Input(shape=(1,))
print(input_tensor)
#%% Ouput
output_layer = Dense(1,name = 'Predicted-Score-Diff')
output_tensor = output_layer(input_tensor)
#%% Model
model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mae', metrics=['accuracy','mae'])
print(model.summary())
plot_model(model, to_file='model.png')
# %%
import matplotlib.pyplot as plt
img = plt.imread('model.png')
plt.imshow(img)
plt.show()
#%%
columns = ['season', 'team1', 'team2', 'home', 'seed_diff', 'score_diff','score_1','score_2','won']

def gen_data(columns,N):
    df = {}
    for column in columns:
        df[column] = np.random.normal(size=int(N))
    return pd.DataFrame(df)

games_tourney = gen_data(columns,4e3)
games_test = games_tourney.sample(frac=0.2)


model.fit(games_tourney['seed_diff'], games_tourney['score_diff'], batch_size=64,  validation_split=0.2, verbose=True, epochs=1)
model.evaluate(games_test['seed_diff'], games_test['score_diff'])
#%%
input_tensor = Input(shape=(1,))
n_teams = 10887
embed_layer = Embedding(input_dim=n_teams, output_dim=1, input_length=1, name='Team-Strength-Lookup')
embed_tensor = embed_layer(input_tensor)
flatten_tensor = Flatten()(embed_tensor)
model = Model(input_tensor, flatten_tensor)
#%%
input_tensor_1 = Input(shape=(1,))
input_tensor_2 = Input(shape=(1,))
shared_layer = Dense(1)
output_tensor_1 = shared_layer(input_tensor_1)
output_tensor_2 = shared_layer(input_tensor_2)

# %%
input_tensor_1 = Input(shape=(1,))
input_tensor_2 = Input(shape=(1,))
output_tensor = Add()([input_tensor_1, input_tensor_2])
input_tensor_3 = Input(shape=(1,))
output_tensor = Add()([input_tensor_1, input_tensor_2, input_tensor_3])
model = Model([input_tensor_1, input_tensor_2, input_tensor_3], output_tensor)
model.compile(optimizer='adam', loss='mae')
model.fit([games_tourney['team1'], games_tourney['team2'], games_tourney['home']], games_tourney['score_diff'], verbose=True, epochs=1)
#%%
input_tensor_1 = Input(shape=(1,))
input_tensor_2 = Input(shape=(1,))
input_tensor_3 = Input(shape=(1,))
output_tensor = Concatenate()([input_tensor_1, input_tensor_2, input_tensor_3])
output_tensor = Dense(1)(output_tensor)
model = Model([input_tensor_1, input_tensor_2, input_tensor_3], output_tensor)
plot_model(model, to_file='model.png')
import matplotlib.pyplot as plt
img = plt.imread('model.png')
plt.imshow(img)
plt.show()
# Other possible is use shared layers for the first two inputs and concatenate the results
# %%
shared_layer = Dense(1)
shared_tensor_1 = shared_layer(input_tensor_1)
shared_tensor_2 = shared_layer(input_tensor_2)
output_tensor = Concatenate()([shared_tensor_1, shared_tensor_2, input_tensor_3])
output_tensor = Dense(1)(output_tensor)
model = Model([input_tensor_1, input_tensor_2, input_tensor_3], output_tensor)
plot_model(model, to_file='model.png')
import matplotlib.pyplot as plt
img = plt.imread('model.png')
plt.imshow(img)
plt.show()
#%%
model = Model([input_tensor_1, input_tensor_2, input_tensor_3], output_tensor)
model.compile(optimizer='adam', loss='mae')
model.summary()
#%%
model.predict([games_tourney['team1'], games_tourney['team2'], games_tourney['home']])
#%%
input_tensor = Input(shape=(1,))
output_tensor = Dense(2)(input_tensor)
model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mae')
model.summary()
X = games_tourney[['seed_diff']]
Y = games_tourney[['score_1', 'score_2']]
model.fit(X, Y, epochs=500,batch_size=16384, verbose=True)
model.get_weights()
#%%
input_tensor = Input(shape=(1,))
output_tensor_reg = Dense(1)(input_tensor)
output_tensor_class = Dense(1, activation='sigmoid', use_bias = False, name='output_tensor_class')(output_tensor_reg)
model = Model(input_tensor, [output_tensor_reg, output_tensor_class])
model.compile(optimizer=Adam(learning_rate = 0.01), loss=['mae', 'binary_crossentropy'])
plot_model(model, to_file='model.png')
import matplotlib.pyplot as plt
img = plt.imread('model.png')
plt.imshow(img)
plt.show()
X = games_tourney[['seed_diff']]
y_reg = games_tourney[['score_diff']]
y_class = games_tourney[['won']]
model.fit(X, [y_reg, y_class], epochs=10, verbose=True)