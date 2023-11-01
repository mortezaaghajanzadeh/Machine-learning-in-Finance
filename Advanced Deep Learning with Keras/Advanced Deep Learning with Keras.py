#%%
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten
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
model.compile(optimizer='adam', loss='mae', metrics=['accuracy','mae'])
print(model.summary())
plot_model(model, to_file='model.png')
# %%
import matplotlib.pyplot as plt
img = plt.imread('model.png')
plt.imshow(img)
plt.show()
#%%
columns = ['season', 'team1', 'team2', 'home', 'seed_diff', 'score_diff','score_1','score_2']

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
