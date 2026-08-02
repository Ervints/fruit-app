# -*- coding: utf-8 -*-
"""
Backbone registry -- swap the pretrained model by changing ONE name
===================================================================
All Keras Applications share the same API, BUT each needs its own
preprocess_input and its own default input size. This registry keeps those
three things together so you can switch models safely.

To use a different backbone, set BACKBONE in fruit_model.py to one of the keys
below and retrain. fruit_detect.py / fruit_app.py read the choice from
model_config.json and apply the matching preprocessing automatically.
"""

from tensorflow.keras.applications import (inception_v3, mobilenet_v2,
                                           resnet50, efficientnet)

# name -> (model class, preprocess_input function, default input size)
BACKBONES = {
    "inceptionv3":    (inception_v3.InceptionV3,   inception_v3.preprocess_input,   299),
    "mobilenetv2":    (mobilenet_v2.MobileNetV2,   mobilenet_v2.preprocess_input,   224),
    "resnet50":       (resnet50.ResNet50,          resnet50.preprocess_input,       224),
    "efficientnetb0": (efficientnet.EfficientNetB0, efficientnet.preprocess_input,  224),
}


def get(name):
    """Return (ModelClass, preprocess_input, img_size) for a backbone name."""
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Options: {list(BACKBONES)}")
    return BACKBONES[name]
