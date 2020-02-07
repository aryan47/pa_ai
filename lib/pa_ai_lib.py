# =============================================================================
# Created by : Ritesh kant
# =============================================================================

import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow import keras
import json

import glob


# stemmer
stemmer = LancasterStemmer()

# set of stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# used to preprocess sentence by tokenizing, removing stop words, stemming


def text_preprocess(sentence, dest):
    # break the sentence in tokens
    words = word_tokenize(sentence)
    # remove stop words and add to train x
    words = [w for w in words if w not in stop_words]
    # stem the word ( handle US and us semantic)
    words = ' '.join([str(stemmer.stem(w.lower())) for w in words])
    dest.append(words)

# Open the application config file and load the configuration


def getAppConfig():
    with open('configuration/app_config.json') as config_data:
        app_config = json.load(config_data)
    return app_config


def createDatasetFromConfig(app_config):
    train_x = []
    train_y = []
    if app_config['train']['files'][0] in {"**", "*.json", "**.json"}:
        location = glob.glob(app_config['train']['dir']+"/**")
    else:
        for files in app_config['train']['files']:
            location.append(files)
   # else:

    raw_train = []
    # gather all the data from provided train location
    for item in location:
        with open(item) as item_data:
            data = json.load(item_data)
            raw_train.extend(data['train_data']['data'])

    # create train x and tain y dataset
    for x in raw_train:
        text_preprocess(x['text'], train_x)
        text_preprocess(x['intent'], train_y)
    return [train_x, train_y]


def createModel(embedding_dim):
    model = keras.Sequential([
        keras.layers.Embedding(100, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # use ReLU in place of tanh function since they are very good alternatives of each other.
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        # Add a Dense layer with 6 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.summary()
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    return model


def tokenize_encode(train, padding=False,):
    tokenizer_x = Tokenizer(num_words=100)
    tokenizer_x.fit_on_texts(train)
    train = np.array(tokenizer_x.texts_to_sequences(train))
    if padding:
        train = keras.preprocessing.sequence.pad_sequences(
            train, padding='post', maxlen=5)
    return (train, tokenizer_x)


def chat(input_text):
    output_value = []
    text_preprocess(input_text, output_value)
    tokenizer_x = Tokenizer(num_words=100)
    tokenizer_x.fit_on_texts([input_text])

    output_value = np.array(tokenizer_x.texts_to_sequences(input_text))
    output = keras.preprocessing.sequence.pad_sequences(
        output_value, padding='post', maxlen=5)
    # output = tokenize_encode(output_value, True)
    model = tf.keras.models.load_model('chatbot')
    ans = model.predict(output)
    print(ans)

    val_max = np.argmax(ans[0])
    for key,val in tokenizer_x.word_index.items():
        if val == val_max:
            k = key

    print(k)

 
def saveTokenizer(token_data):
    token_data = token_data.get_config()
    with open('tokenized_data.json', 'w') as handle:
        json.dump((token_data), handle)

