# Created by : Ritesh kant

import tensorflow as tf
from lib.pa_ai_lib import getAppConfig, createDatasetFromConfig, createModel, tokenize_encode


config = getAppConfig()
train_x, train_y = createDatasetFromConfig(config)
model = createModel(5)

train_x = tokenize_encode(train_x, True)
train_y = tokenize_encode(train_y)

model.fit(train_x,train_y, epochs=30)





