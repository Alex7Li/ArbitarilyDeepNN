import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
console = Console()

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# https://www.tensorflow.org/tutorials/keras/regression
def download_dataset(dataset_name):
    assert dataset_name == 'autompg'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                            na_values='?', comment='\t',
                            sep=' ', skipinitialspace=True)
    dataset = raw_dataset.copy().dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')
    return train_features, train_labels, test_features, test_labels

def make_linear_model(train_features):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(units=1)
    ])
    linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mean_absolute_error')
    return linear_model

class MyModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

def make_inf_model(train_features):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    embed_size = 32
    input_size = 32
    total_size = 256
    output_size = 32
    input = normalizer(tf.keras.layers.Input(shape=train_features.shape[1],))
    batch_size = tf.shape(input)[0]
    assert input_size + output_size <= total_size
    first_layer_value = layers.Dense(input_size * embed_size, activation='relu')
    first_layer_query = layers.Dense(input_size * embed_size, activation='relu')

    random_init_query = tf.Variable(tf.random.normal((1, total_size - input_size, embed_size), 0, 1))
    random_init_query = tf.repeat(random_init_query, batch_size, axis=0)
    random_init_value = tf.Variable(tf.random.normal((1, total_size - input_size, embed_size), 0, 1))
    random_init_value = tf.repeat(random_init_value, batch_size, axis=0)
    flatten = tf.keras.layers.Flatten()

    inf_layer = layers.Attention(total_size, dropout=.5)
    last_layer = layers.Dense(output_size * embed_size, activation='relu')

    input_query = tf.reshape(first_layer_value(input), (-1, input_size, embed_size))
    input_value = tf.reshape(first_layer_query(input), (-1, input_size, embed_size))
    mid_query = tf.concat([input_query, random_init_query], axis=1)
    mid_value = tf.concat([input_value, random_init_value], axis=1)
    output_embeds = []
    for _ in range(1):
        mid_value = inf_layer([mid_query, mid_value])
        output_embed = mid_value[:, total_size - output_size:, :] 
    output = last_layer(flatten(output_embed))#tf.reduce_mean(output_embeds, axis=0))
    
    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mean_absolute_error')
    return model

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

if __name__ == "__main__":
    train_features, train_labels, test_features, test_labels = download_dataset('autompg')
    if False:
        model = make_linear_model(train_features)
        model_name = 'linear_model'
    else:
        model = make_inf_model(train_features)
        model_name = 'inf_linear_model'
    filepath = f'weights/{model_name}'
    # try:
    #     model.load_weights(filepath)
    # except tf.errors.NotFoundError:
    #     console.print("No weights to load", style="bold red")
    # except ValueError:
    #     console.print("Incompatible Model", style="bold red")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)
    history = model.fit(train_features, train_labels, epochs=200,
        validation_split = 0.2, callbacks=[cp_callback])
    plot_loss(history)
    test_results = {}
    test_results[model_name] = model.evaluate(
        test_features, test_labels)
    
    print(test_results)

    



