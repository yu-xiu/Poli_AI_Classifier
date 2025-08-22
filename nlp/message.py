from google.colab import drive
import pandas as pd
import spacy

from sklearn.neighbors import KNeighborsClassifier
drive.mount('/content/drive', force_remount=True)

from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""# Preprocessing"""

from sklearn.preprocessing import LabelEncoder

file_path = '/content/drive/Shareddrives/CS_273/political_social_media.csv'

res_df = pd.read_csv(file_path , encoding='ISO-8859-1')
res_df.head()

res_df = res_df[res_df['message'] != 'other']

def safe_sample(g):
    n_samples = min(len(g), 200)
    return g.sample(n=n_samples, replace=False)

#res_df = res_df.groupby('message').apply(safe_sample).reset_index(drop=True)

res_df = res_df.groupby('message').apply(lambda x: x.sample(n=200, replace=True)).reset_index(drop=True)

text_list = res_df["text"].tolist()

class_ = res_df["message"]
encoder = LabelEncoder()

encoded_class = encoder.fit_transform(class_)
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))



cleaned_text_list = []
import re

from html import unescape

for text in text_list:
  clean = re.sub(r'\\x[0-9A-Fa-f]{2}', '', text)
  clean = re.sub(r'[^\x00-\x7F]+', '', clean)
  pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
  clean = re.sub(pattern, '', clean)
  clean = unescape(clean)

  cleaned_text_list.append(clean)

import collections

print(cleaned_text_list)
print((encoded_class))
print(collections.Counter(encoded_class))
print(label_mapping)

#Splitting datasets in to train test
from sklearn.model_selection import train_test_split
def split_data(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .8, shuffle = True, random_state=42)

  return X_train, y_train, X_test, y_test

#train model and report test and train evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, f1_score

def train_model(model, X_train, y_train, X_test, y_test, X_valid, y_valid):
  model.fit(X_train, y_train)

  # Make predictions on the test data
  y_pred = model.predict(X_test)

  # Calculate metrics for test and train
  test_accuracy = accuracy_score(y_test, y_pred)
  test_f1 = f1_score(y_test, y_pred, average='macro')



  classification_report_ = classification_report(y_test, y_pred, target_names=['0', '1', '2', '3', '4', '5', '6', '7'])

  return test_accuracy, test_f1, classification_report_, model

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(documents, ngram_range_):
  tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range_)
  tfidf_vectorizer.fit(documents)

  vectors = tfidf_vectorizer.transform(documents)
  columns_ = tfidf_vectorizer.get_feature_names_out()
  tf_idf_df = pd.DataFrame(data=vectors.toarray(), columns=columns_)

  return tf_idf_df, vectors.toarray()

def tf_vecotrize(cleaned_text_list):
  tf_df, tf_vectors = tf_idf(cleaned_text_list, (1,3))
  return tf_df, tf_vectors

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

# remove stop words
def remove_stop_words(documents):
  stop_words = set(stopwords.words('english'))
  res = []
  for document in documents:
    sent = []
    for word in document.split():
      if word.lower() not in stop_words:
        sent.append(word.lower())
    res.append(' '.join(sent))

  return res

#stemming
def stem_documents(documents):
  stemmer = PorterStemmer()

  res = []
  for document in documents:
    sent = []
    for word in document.split():
      sent.append(stemmer.stem(word.lower()))
    res.append(' '.join(sent))

  return res

import spacy
def lemmatization(documents):
  nlp = spacy.load("en_core_web_sm")

  #print(documents)
  tokenized_text = []

  for document in documents:
    tokenized_text.append(nlp(document))

  lem_text = []

  for sent in tokenized_text:
    lem_sentence = [token.lemma_ for token in nlp((sent))]
    #print(lem_sentence)
    lem_text.append(' '.join(lem_sentence))

  return lem_text

# no stop words and with lemmatization
document_no_stops = remove_stop_words(cleaned_text_list) # no stop words
document_no_stops_lemma = lemmatization(document_no_stops) # no stop words with lemmatization

"""# TF-IDF Multinomial Naive Bayes Classifier - Model 1"""

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report


def mb(X_train, y_train, X_test, y_test):
  # Initialize the Multinomial Naive Bayes classifier
  model = MultinomialNB()

  # Train the classifier
  model.fit(X_train, y_train)


  # Predict the labels for the test data
  y_pred = model.predict(X_test)

  #Metrics
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy}")
  f1 = f1_score(y_test, y_pred, average='micro')

  print(f'The F1 score is: {f1}')
  report = classification_report(y_test, y_pred)


  print(report)

  return model

"""##Fine Tuning"""

import warnings

def hyperparameter_tuning(param_grid, model, X_train, y_train, X_test):
  warnings.filterwarnings('ignore')

  scorer = make_scorer(f1_score, average='micro') #creates f1_score


  param_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer)
  param_search.fit(X_train, y_train)

  return param_search.best_params_, param_search.best_score_, param_search.best_estimator_

def print_metrics_after_ht(best_params_, best_score_):
  print(f"The best parameters are: {best_params_}")
  print(f"The corresponding best f1_score is {best_score_}")

"""NO PREPROCESSING"""

from joblib import dump, load


param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0]
}

tf_df, tf_vectors = tf_vecotrize(cleaned_text_list)
X_train, y_train, X_test, y_test = split_data(tf_df, encoded_class)
'''print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)'''

X_train.head(3)

model = mb(X_train, y_train, X_test, y_test)
bp, bs, optimal_model = hyperparameter_tuning(param_grid, model, X_train, y_train, X_test)
print_metrics_after_ht(bp, bs)

dump({'model': optimal_model, 'parameters': bp}, '/content/drive/Shareddrives/CS_273/MESSAGE_naive_bayes.joblib')

"""# BERT - Model 2"""

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd

def bert_ft(x_list, y):

  #y = (res_df["positive=1/negative=0"]).tolist()
  X = x_list
  X_train, y_train, X_test, y_test = split_data(X, y)

  #print(X_train)
  #print(type(X_train))
  #print(y_train)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

  max_len = 0
  for review in X_train:
    max_len = max(max_len, len(review))

  #print(max_len)

  max_len = 256

  # Tokenize and encode the sentences
  X_train_encoded = tokenizer.batch_encode_plus(X_train,
                                                padding=True,
                                                truncation=True,
                                                max_length = max_len,
                                                return_tensors='tf')

  print(type(X_train_encoded))



  X_test_encoded = tokenizer.batch_encode_plus(X_test,
                                                padding=True,
                                                truncation=True,
                                                max_length = max_len,
                                                return_tensors='tf')


  #print(X_test_encoded)

  #FineTuning BERT

  model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)

  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
  model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


  history = model.fit(
      [X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']],
      pd.Series(y_train),
      validation_data=(
        [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']],pd.Series(y_test)),
      batch_size=8,
      epochs=8
  )


  test_loss, test_accuracy = model.evaluate(
      [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']],
      pd.Series(y_test)
  )

  #print(test_loss, test_accuracy)


  from sklearn.metrics import f1_score

  raw_predictions = model.predict([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']])
  #print(raw_predictions)

  import numpy as np

  logits = (raw_predictions.logits)
  #print(logits)
  y_test_pred = []
  for logit in logits:
    x = np.argmax(logit)
    y_test_pred.append(x)
    #print(x)





  f1 = f1_score(y_test, y_test_pred, average='micro')

  print(f"BERT F1: {f1}")
  return model

x_list = cleaned_text_list
model = bert_ft(x_list, encoded_class.tolist())

model.save_pretrained("/content/drive/Shareddrives/CS_273/MESSAGE_BERT")

"""# GPT2 - Model 3"""

from transformers import AutoTokenizer, TFGPT2ForSequenceClassification, TFGPT2Model
import tensorflow as tf
from transformers import GPT2Model, GPT2Tokenizer

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def gpt2_ft(x_list, y):




  X = x_list
  x_train, y_train, x_test, y_test = split_data(X, y)

  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  tokenizer.pad_token = tokenizer.eos_token
  model = TFGPT2Model.from_pretrained("gpt2")

  # Tokenize input data
  train_encodings = tokenizer(x_train, truncation=True, padding=True, return_tensors="tf")
  test_encodings = tokenizer(x_test, truncation=True, padding=True, return_tensors="tf")

  # Define input layer
  input_ids = Input(shape=(None,), dtype='int32', name='input_ids')
  attention_mask = Input(shape=(None,), dtype='int32', name='attention_mask')

  # GPT-2 output
  output = model(input_ids, attention_mask=attention_mask)

  # Add classification layer with two neurons for softmax
  classification_output = Dense(8, activation='softmax')(output[0][:, -1, :])

  # Create model
  classification_model = Model(inputs=[input_ids, attention_mask], outputs=[classification_output])

  # One-hot encode labels
  y_train_encoded = to_categorical(y_train, num_classes=8)
  y_test_encoded = to_categorical(y_test, num_classes=8)

  # Compile the model
  classification_model.compile(optimizer=Adam(learning_rate=5e-5), loss='categorical_crossentropy', metrics=['accuracy'])

  # Fit the model
  classification_model.fit({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},
                          y_train_encoded,
                          epochs=8,
                          batch_size=8)


  '''test_loss, test_accuracy = model.evaluate(
      {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']}, y_test_encoded

  )'''

  #print(test_loss, test_accuracy)


  from sklearn.metrics import f1_score

  raw_predictions = classification_model.predict({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']})
  #print(raw_predictions)

  #print(raw_predictions)
  #logits = (raw_predictions.logits)
  #print(logits)
  import numpy as np
  y_test_pred = []
  for logit in raw_predictions:
    x = np.argmax(logit)
    y_test_pred.append(x)



  f1 = f1_score(y_test, y_test_pred, average='micro', pos_label=1)

  print(f"GPT2 F1: {f1}")
  return classification_model,tokenizer

x_list = cleaned_text_list
model, tokenizer = gpt2_ft(x_list, encoded_class.tolist())



"""#Long Short Term Memory (LSTM) - Model 4"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""## Get padded sequences for reviews"""

def get_sequences_of_reviews(reviews):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(reviews) # learn a vocabulary of all unique words in the review text data and assigned a unique integer ID to each word
  reviews_word_index = tokenizer.word_index # word : index dict
  sequences_of_reviews = tokenizer.texts_to_sequences(reviews) # convert review text to sequence of numerical values, those numbers are form the word_index numbers assigned to the words
  max_sequences_length = max(len(seq) for seq in sequences_of_reviews)
  padded_sequences = pad_sequences(sequences_of_reviews, maxlen=max_sequences_length, padding="post") # ensure all sequences have the same length by padding 0 to the sequences that are shorter than the max_sequence_length
  return padded_sequences

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def create_lstm_model(X, num_units=64, learning_rate=0.001):
  model = keras.Sequential()

  MAX_NB_WORDS = 25000
  EMBEDDING_DIM = 50


  print(X.shape[1])
  model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
  model.add(SpatialDropout1D(0.2))
  model.add(keras.layers.Bidirectional(keras.layers.LSTM(num_units)))
  model.add(keras.layers.Dense(36, activation='relu'))
  model.add(Dense(8, activation='sigmoid'))

  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

from tensorflow.keras.callbacks import EarlyStopping
def train_lstm(X_train, y_train):
  model = create_lstm_model(X=X_train)
  epochs = 20
  batch_size = 64

  from keras.callbacks import ModelCheckpoint
  checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
  early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
  history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[checkpoint, early_stopping]) # best model history

  return history, model # return the model with best performance

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from tensorflow.keras.utils import to_categorical



def model_training(text, encoded_class):

  X = get_sequences_of_reviews(text)
  Y = encoded_class
  print(type(Y))



  X_train, y_train, X_test, y_test = split_data(X, np.array(Y))

  y_train_one_hot = to_categorical(y_train, num_classes=8)
  y_test_one_hot = to_categorical(y_test, num_classes=8)
  history, model = train_lstm(X_train, y_train_one_hot)
  print(X_train.shape,y_train.shape)
  print(X_test.shape,y_test.shape)
  print(X_test)

  #model.load_weights('best_model.h5')
  #model.save_weights('/content/drive/Shared drives/CS_273/capstone_best_model_weights_11_23.h5')
  model.save('/content/drive/Shared drives/CS_273/message_capstone_best_lstm_model_11_23.h5')
  return model, X_test, y_test_one_hot

def compute_test_f1(model, X_test, y_test_one_hot):
  y_pred = model.predict(X_test)
  # Evaluate the model's performance
  from sklearn.metrics import precision_recall_fscore_support, accuracy_score
  accuracy = accuracy_score(y_test_one_hot, y_pred.round())
  precision, recall, f1, _ = precision_recall_fscore_support(y_test_one_hot, y_pred.round(), average='micro')
  return f1, accuracy

model, X_test, y_test_one_hot = model_training(cleaned_text_list, encoded_class.tolist())

f1, accuracy = compute_test_f1(model, X_test, y_test_one_hot)
print("LSTM test f1 score for no preprocessing data", f1)