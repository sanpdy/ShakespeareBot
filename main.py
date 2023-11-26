import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

characters = sorted(set(text))

charToIndex = dict((c,i) for i, c in enumerate(characters))
indexToChar = dict((i,c) for i, c in enumerate(characters))

SEQUENCE_LENGTH = 40
STEP_SIZE = 3

sentences = []
nextCharacters = []

# Use this to train the model
'''
for i in range(0, len(text) - SEQUENCE_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQUENCE_LENGTH])
    nextCharacters.append(text[i+SEQUENCE_LENGTH])

x = np.zeros((len(sentences), SEQUENCE_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, charToIndex[character]] = 1
        y[i, charToIndex[nextCharacters[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.01))
model.fit(x,y,batch_size=256, epochs=4)
model.save('textgenerator.model')
'''

# Load model once it's trained
model = tf.keras.models.load_model('textgenerator.model')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generateText(length, temperature):
    start_index = random.randint(0, len(text) - SEQUENCE_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQUENCE_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQUENCE_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, charToIndex[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                                 temperature)
        next_character = indexToChar[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

print('--------Temperature = 0.2 --------')
print(generateText(300, 0.2))

print('--------Temperature = 0.4 --------')
print(generateText(300, 0.4))

print('--------Temperature = 0.6 --------')
print(generateText(300, 0.6))

print('--------Temperature = 0.8 --------')
print(generateText(300, 0.8))

print('--------Temperature = 1 --------')
print(generateText(300, 1))
