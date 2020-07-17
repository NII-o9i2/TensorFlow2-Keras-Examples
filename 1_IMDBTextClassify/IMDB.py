# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import tensorflow as tf 
import tensorflow_hub as hub 
import tensorflow_datasets as tfds


# %%

from keras.datasets import imdb
(x_train,y_train),(x_test,y_test) = imdb.load_data(path = "imdb.npz",
num_words=10000,
skip_top=0,
maxlen=None,
seed=113,
start_char=1,
oov_char=2,
index_from=3)
'''

train_data,validation_data,test_data = tfds.load(
    name = "imdb_reviews",
    split=('train[:60%]','train[60%:]','test'),
    as_supervised=True
)
'''


# %%
word_index = imdb.get_word_index()
# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
print(decoded_review)


# %%
import numpy as np 

def one_hot_encoder(sequences,dimension = 10000):
    results = np.zeros((len(sequences),dimension))
    for i,j in enumerate(sequences):
        results[i,j] = 1.
    return results


# %%
OH_x_train = one_hot_encoder(x_train)
OH_x_test = one_hot_encoder(x_test)


# %%
y_train_v = np.asarray(y_train).astype('float32')
y_test_v = np.asarray(y_test).astype('float32')

# %% [markdown]
# # use One-hot

# %%
from keras import models
from keras import layers

model = models.Sequential()
model.reset_states()

model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation = 'sigmoid'))
model.build(input_shape=(None,10000))
model.summary()

# %% [markdown]
# # use keras.Embedding

# %%
from keras.preprocessing import sequence
maxword = 400 
x_train_embedding = sequence.pad_sequences(x_train,maxlen=maxword)
x_test_embedding = sequence.pad_sequences(x_test,maxlen=maxword)
print(x_train_embedding[0],"len is ",x_test_embedding[0].size)
vocab_size = np.max([np.max(x_train_embedding[i]) for i in range(x_train_embedding.shape[0])]) + 1

print(vocab_size)
model = models.Sequential()
model.add(layers.Embedding(10000,64,input_length=400))
model.add(layers.Flatten())
model.add(layers.Dense(500,activation='relu'))
model.add(layers.Dense(500,activation='relu'))
model.add(layers.Dense(200,activation='relu'))
model.add(layers.Dense(50,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.build(input_shape=(None,None))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 
print(model.summary())

# %% [markdown]
# # function debug

# %%
import keras
x1 = keras.layers.Input(shape=(None,400))
layer2 = layers.Embedding(10000,64,input_length=400)
layer3 = layers.Dense(5,activation='relu')
inputs = tf.constant(x_train_embedding[:5])
#print(inputs)
yy = layer3(inputs)
print(yy[0])
yy = layer2(inputs)
print(yy)


# %%
from tensorflow.keras import layers

x3 = layers.Dense(32, activation='relu')
inputs = tf.random.uniform(shape=(10, 20))

outputs = x3(inputs)


# %%
trainx = x_train_embedding
trainy = y_train
testx = x_test_embedding
testy = y_test
'''
trainx = x_train.astype('float64')
trainx = y_train.astype('float64')
testx = x_test.astype('float64')
testy = y_test.astype('float64')
'''
model.fit(trainx,trainy,validation_data=(testx,testy), epochs = 20,batch_size = 100, verbose = 1)
score = model.evaluate(testx,testy)


# %%
print(score)


# %%


