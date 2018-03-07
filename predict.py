import os
import  numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

width, height = 64, 64
model_p = 'model.h5'
weight_p = 'weights.h5'

model = load_model(model_p)
model.load_weights(weight_p)

tmg_path = 'dataset/single_prediction/cat_or_dog_2.jpg'


def fpredict(file):
    test_image = load_img(file, target_size=(width,height))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    array = model.predict(test_image)
    result = array[0]
    if result[0][0] == 1:
        an = 'Dog'
    else:
        an = 'Cat'
    return an
res = fpredict(tmg_path)
print(res)