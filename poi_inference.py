from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

img_path = 'data/test_img/black_casual_dress.jpeg'

poi_model = load_model('data/poi_model_checkpoint_1.h5')
poi_model.compile(optimizer='adadelta', loss='mean_squared_error')


with open(img_path, 'r+b') as f:
    with Image.open(f) as photo:
        pic = photo.resize((150, 150), Image.ANTIALIAS)
        x = np.array(pic)
        x = (x - 255) / 255
        x = np.expand_dims(x, axis=0)
        prediction = poi_model.predict(x)[0] * 100
        print(prediction)
