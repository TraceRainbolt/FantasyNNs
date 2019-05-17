import sys

import numpy as np
from keras import models, layers
from keras.utils import normalize
from statsmodels import robust

from keras import backend as K


EPOCHS = 5
BATCH_SIZE = 32

POINT_INDEX = 2

def main():
    position = sys.argv[1] if len(sys.argv) > 1 else 'all'
    dataset = np.load(f'{position}_dataset.npy')
    print(dataset.shape)
    train(position, dataset)


def train(position, dataset):
    label_column = dataset.shape[1] - 1
    input_data = dataset[:, 1:]
    targets = dataset[:, 0]

    input_data = normalize(
        input_data.astype('float32'),
        axis=-1,
    )

    model = models.Sequential()
    model.add(layers.Dropout(0.25, input_shape=(input_data.shape[1],)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    print(model.summary())

    model.compile(optimizer='adam', loss='mae',
                  metrics=['mae'])

    model.fit(
        x=input_data,
        y=targets,
        validation_split=.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,)


    model.save(f'{position}_player_fp_predict.h5')


if __name__ == '__main__':
    main()
