import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

vgg = VGG16(
    input_shape=(244,244,3),
    weights='imagenet',
    include_top=False,  # no final three layers: pooling, dropout and dense
    classes=None
)

print(vgg.summary())