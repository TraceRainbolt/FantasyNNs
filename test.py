from keras import models
import numpy as np
from keras.utils import normalize

dataset = np.load('dataset.npy')
np.random.shuffle(dataset)

inputs = dataset[:, 1: ]
targets = dataset[:, 0].astype('float64')

input_data = normalize(
    inputs.astype('float32'),
    axis=-1,
)

offenseive_model = models.load_model('offensive_player_fp_predict.h5')

prediction = offenseive_model.predict(input_data)

print('  Target   Predicted')

total = 0
for i in range(input_data.shape[0])[:20]:
    predicted = prediction[i][0]
    print(inputs[i])
    print(f'{targets[i]:6.2f} \t {predicted:6.2f}')