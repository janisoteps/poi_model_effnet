# import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint
import jsonlines
import numpy as np
from batch_generator import DataGenerator
from tensorflow.python.keras.applications.efficientnet import EfficientNetB6

input_data = '/Users/janis/dev/garms_experiment/poi_trainer/data/poi_train_data_1.jsonl'
params = {
    'dim_x': 150,
    'dim_y': 150,
    'batch_size': 16,
    'shuffle': True
}
batch_size = 16


def load_ids(data_file_path):
    print('Loading ids and labels from data file')
    path_dict = {'train': [], 'validation': []}
    label_dict = {}
    line_counter = 0
    with jsonlines.open(data_file_path, 'r') as infile:
        for line in infile:
            img_path = line['path']  # replacing with path to enable img retrieval from multiple folders
            label = list(line['label'])
            label_dict[img_path] = label
            train_prob = np.random.random() * 10
            print('Line LOADED: ', str(line_counter))
            print('Image path loaded: ', img_path)
            line_counter += 1

            if train_prob > 1:
                path_dict['train'].append(img_path)
            else:
                path_dict['validation'].append(img_path)

    return path_dict, label_dict


partition, labels = load_ids(input_data)
# Generators
training_generator = DataGenerator(**params).generate(labels, partition['train'])
validation_generator = DataGenerator(**params).generate(labels, partition['validation'])

efficientnet_base = EfficientNetB6(
    include_top=False,
    weights="imagenet",
    input_shape=(150, 150, 3)
)
# efficientnet_base.trainable = False
for i, layer in enumerate(efficientnet_base.layers):
    if i < 645:
        layer.trainable = False
    else:
        print(layer.name)
        layer.trainable = True

dropout_rate = 0.2
poi_model = models.Sequential()
poi_model.add(efficientnet_base)
poi_model.add(layers.GlobalMaxPooling2D(name="gap"))

if dropout_rate > 0:
    poi_model.add(layers.Dropout(dropout_rate, name="dropout_out"))

poi_model.add(layers.Dense(2, activation="sigmoid", name="fc_out"))

poi_model.compile(optimizer='adadelta', loss='mean_squared_error')
print(poi_model.summary())

checkpoint_path = 'poi_model_checkpoint_1.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint]

history = poi_model.fit_generator(
    generator=training_generator,
    steps_per_epoch=len(partition['train'])//batch_size,
    validation_data=validation_generator,
    validation_steps=len(partition['validation'])//(batch_size * 2),
    callbacks=callback_list,
    epochs=10
)

poi_model.save('data/poi_model_1.h5')
