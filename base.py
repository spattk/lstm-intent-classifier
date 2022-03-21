
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.python.keras.layers import (LSTM, Activation, Bidirectional,
                                            Dense, Dropout, Embedding, Input)
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

data_full_json_path = 'data/data_full.json'
glove_path = 'utils/glove.6B.100d.txt'

with open(data_full_json_path) as file:
  data = json.loads(file.read())
  
def get_train_test_split(text, labels, t_size):
  return train_test_split(text, labels, test_size=t_size)

def get_train_test_encoded(train_labels, test_labels):
  label_encoder = LabelEncoder()
  integer_encoded = label_encoder.fit_transform(classes)
  onehot_encoder = OneHotEncoder(sparse=False)
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  onehot_encoder.fit(integer_encoded)
  train_labels_encoded = label_encoder.transform(train_labels)
  train_labels_encoded = train_labels_encoded.reshape(len(train_labels_encoded), 1)
  train_labels = onehot_encoder.transform(train_labels_encoded)
  test_labels_encoded = label_encoder.transform(test_labels)
  test_labels_encoded = test_labels_encoded.reshape(len(test_labels_encoded), 1)
  test_labels = onehot_encoder.transform(test_labels_encoded)
  return train_labels, train_labels_encoded, test_labels, test_labels_encoded, label_encoder

def compute_train_test_sequences(train_txt, test_txt, max_num_words):
  tokenizer = Tokenizer(num_words=max_num_words)
  tokenizer.fit_on_texts(train_txt)
  train_sequences = tokenizer.texts_to_sequences(train_txt)
  train_sequences = pad_sequences(train_sequences, maxlen=maxLen, padding='post')
  test_sequences = tokenizer.texts_to_sequences(test_txt)
  test_sequences = pad_sequences(test_sequences, maxlen=maxLen, padding='post')
  word_index = tokenizer.word_index
  return train_sequences, test_sequences, word_index, tokenizer

def get_text_labels(data):
  validation_oos = np.array(data['oos_val'])
  train_oos = np.array(data['oos_train'])
  test_oos = np.array(data['oos_test'])
  validation_rest = np.array(data['val'])
  train_rest = np.array(data['train'])
  test_rest = np.array(data['test'])
  validation_data = np.concatenate([validation_oos,validation_rest])
  train_data = np.concatenate([train_oos,train_rest])
  test_data = np.concatenate([test_oos,test_rest])
  data = np.concatenate([train_data,test_data,validation_data])
  data = data.T
  text = data[0]
  labels = data[1]
  return text, labels

def get_max_length_text(train_txt, percentile):
  ls=[]
  for c in train_txt:
      ls.append(len(c.split()))
  return int(np.percentile(ls, percentile))

def compute_embedding_matrix(embeddings_index, max_num_words, word_index, embedding_dim, emb_mean, emb_std):
  num_words = min(max_num_words, len(word_index) )+1
  embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embedding_dim))
  for word, i in word_index.items():
      if i >= max_num_words:
          break
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector
  return embedding_matrix, num_words

def compute_embeddings_index(glove_path, enc):
  embeddings_index={}
  with open(glove_path, encoding=enc) as f:
      for line in f:
          values = line.split()
          word = values[0]
          coefs = np.asarray(values[1:], dtype='float32')
          embeddings_index[word] = coefs
  return np.stack(embeddings_index.values()), embeddings_index

def get_intent_classifier_model(num_words, train_sequences, embedding_matrix, classes):
  model = Sequential()
  model.add(Embedding(num_words, 100, trainable=False,input_length=train_sequences.shape[1], weights=[embedding_matrix]))
  model.add(Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.1, dropout=0.1), 'concat'))
  model.add(Dropout(0.3))
  model.add(LSTM(256, return_sequences=False, recurrent_dropout=0.1, dropout=0.1))
  model.add(Dropout(0.3))
  model.add(Dense(50, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(classes.shape[0], activation='softmax'))
  return model

def draw_plots(history):
  %matplotlib inline
  
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show()

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show()

text, labels = get_text_labels(data)
train_txt,test_txt,train_labels,test_labels = get_train_test_split(text, labels, 0.3)
maxLen = get_max_length_text(train_txt, 98)
all_embeddings, embeddings_index = compute_embeddings_index(glove_path, 'utf-8')
embedding_dim = len(embeddings_index['the'])
emb_mean,emb_std = all_embeddings.mean(), all_embeddings.std()
classes = np.unique(labels)
train_sequences, test_sequences, word_index, tokenizer = compute_train_test_sequences(train_txt, test_txt, 40000)
embedding_matrix, num_words = compute_embedding_matrix(embeddings_index, 40000, word_index, embedding_dim, emb_mean, emb_std)
train_labels, train_labels_encoded, test_labels, test_labels_encoded, label_encoder = get_train_test_encoded(train_labels, test_labels)

model = get_intent_classifier_model(num_words, train_sequences, embedding_matrix, classes)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(train_sequences, train_labels, epochs = 20, batch_size = 64, shuffle=True, validation_data=[test_sequences, test_labels])
draw_plots(history)

model.save('models/intents.h5')
with open('utils/classes.pkl','wb') as file:
   pickle.dump(classes,file)

with open('utils/tokenizer.pkl','wb') as file:
   pickle.dump(tokenizer,file)

with open('utils/label_encoder.pkl','wb') as file:
   pickle.dump(label_encoder,file)

class IntentClassifier:
    def __init__(self,classes,model,tokenizer,label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return label_encoder.inverse_transform(np.argmax(self.pred,1))[0]
    
    
model = load_model('models/intents.h5')
with open('utils/classes.pkl','rb') as file:
  classes = pickle.load(file)
  
with open('utils/tokenizer.pkl','rb') as file:
  tokenizer = pickle.load(file)
  
with open('utils/label_encoder.pkl','rb') as file:
  label_encoder = pickle.load(file)
  
nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
nlu.get_intent("what's the weather outsie")


