import numpy as np
from keras import models, layers
from keras.utils import normalize

EPOCHS = 200
BATCH_SIZE = 64

POINT_INDEX = 2

NUM_FEATURES = 20


def main():
    dataset = np.load('dataset.npy')
    train(dataset)

def train(dataset):
    # input_data = normalize(
    #     x.astype('float32'),
    #     axis=-1,
    # )
    model = models.Sequential()
    model.add(layers.LSTM(128, return_sequences=True, input_shape=(None, NUM_FEATURES)))
    model.add(layers.LSTM(128, return_sequences=True,))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(32))
    model.add(layers.Dense(1))
    print(model.summary())

    model.compile(optimizer='adam', loss='mae',
                  metrics=['mae'])

    history = model.fit_generator(
        generate(dataset),
        steps_per_epoch=len(dataset) * 10 // BATCH_SIZE,
        epochs=EPOCHS)

    model.save('offensive_player_fp_predict.h5')


def generate(dataset):
    n = 5

    batch_x = np.zeros((BATCH_SIZE, n, NUM_FEATURES))
    batch_y = np.zeros((BATCH_SIZE, 1))

    samples = 0
    count = 0
    while True:
        for player_games in dataset:
            # Not enough data, move along
            if len(player_games) <= n:
                continue

            subsets = len(player_games) // n

            for i in range(0, subsets, n):
                batch_x[samples] = player_games[i:i + n]
                batch_y[samples] = player_games[i + n][2]
                samples += 1
                if samples == BATCH_SIZE:
                    print(batch_x)
                    yield batch_x, batch_y
                    batch_x = np.zeros((BATCH_SIZE, n, NUM_FEATURES))
                    batch_y = np.zeros((BATCH_SIZE, 1))
                    samples = 0

if __name__ == '__main__':
    main()
