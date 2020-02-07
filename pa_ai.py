# =============================================================================
# Created by : Ritesh kant
# =============================================================================

import tensorflow as tf

from lib.pa_ai_lib import getAppConfig, createDatasetFromConfig, createModel, tokenize_encode, chat, saveTokenizer

config = getAppConfig()
train_x, train_y = createDatasetFromConfig(config)
model = createModel(5)

train_x, tokenizer_x = tokenize_encode(train_x,padding= True)
train_y, tokenizer_y = tokenize_encode(train_y)

saveTokenizer(tokenizer_y)

# model.fit(train_x, train_y, epochs=30)
# model.save('chatbot') 

in_val = input('You: ')
chat(in_val)
