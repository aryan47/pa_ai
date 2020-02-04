# Created by : Ritesh kant

import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize

import numpy
import tensorflow
from tensorflow import keras
import json

import glob

# stores the training dataset
raw_train = []

# stores training x
train_x = []

# stores training y
train_y = []

# stemmer
stemmer = LancasterStemmer()

# set of stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# used to preprocess sentence
def text_preprocess(sentence, dest):
    # break the sentence in tokens
    words = word_tokenize(sentence)
    # remove stop words and add to train x
    words = [w for w in words if w not in stop_words]
    # stem the word ( handle US and us semantic)
    words = ' '.join([str(stemmer.stem(w.lower())) for w in words])
    dest.append(words)


with open('configuration/app_config.json') as config_data:
    app_config = json.load(config_data)

if app_config['train']['files'][0] in {"**", "*.json", "**.json"}:
    location = glob.glob(app_config['train']['dir']+"/**")
else:
    for files in app_config['train']['files']:
        location.append(files)
   # else:

# gather all the data from provided train location
for item in location:
    with open(item) as item_data:
        data = json.load(item_data)
        raw_train.extend(data['train_data']['data'])

# create train x and tain y dataset
for x in raw_train:
    text_preprocess(x['text'], train_x)
    text_preprocess(x['intent'],train_y)


print(train_y)

