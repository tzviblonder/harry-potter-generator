#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.layers import Embedding,LSTM,GRU,Dense
from tensorflow.keras import Input
import tensorflow_probability as tfp
import docx
import random

filepath = '../input/harry-potter/complete_harry_potter.docx'

doc = docx.Document(filepath)

full_text = []

for paragraph in doc.paragraphs:
    text = paragraph.text
    if text.isupper() == False and 'J.K. Rowling' not in text and text != '':
        full_text.append(text)
full_text = '\n'.join(full_text)
full_text = full_text.replace('\n','')


# In[2]:


unique_characters = sorted(set(full_text))
vocab_size = len(unique_characters)
print('There are {} unique characters.'.format(vocab_size))

char_tokenizer = keras.layers.StringLookup(vocabulary=unique_characters)

detokenizer = keras.layers.StringLookup(vocabulary=char_tokenizer.get_vocabulary(),
                                                                  invert=True)

split_text = tf.strings.unicode_split(full_text,'UTF-8')
tokenized_text = char_tokenizer(split_text)


# ### The data will be sequences of characters of a specified length (defined in the variable sequence_length). The input will be the entire sequence besides the final character, and the output is the entire sequence besides the first character. The shift variable defines how many characters are between the beginning of one sequence and the next, so a smaller shift leads to a larger dataset.

# In[3]:


sequence_length = 131
batch_size = 100
shuffle_size = 100
shift = 14

def make_dataset(tokenized_text,shift=shift):
    dataset = tf.data.Dataset.from_tensor_slices(tokenized_text)
    dataset = dataset.window(sequence_length,
                             shift=shift,
                             drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(sequence_length))
    dataset = dataset.map(lambda window: (window[:-1],window[1:]))
    dataset = dataset.batch(batch_size,drop_remainder=True).shuffle(shuffle_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
dataset = make_dataset(tokenized_text)


# In[4]:


print('Examples of inputs and outputs:\n')
t = 16
for batch in iter(dataset):
    if random.randint(0,300) == 1:
        num = random.randint(0,batch_size-1)
        input_example = tf.strings.reduce_join(detokenizer(batch[0][num]),
                                             axis=-1).numpy().decode()
        output_example = tf.strings.reduce_join(detokenizer(batch[1][num]),
                                             axis=-1).numpy().decode()
        print('Input example:\n',input_example)
        print('\nOutput example:\n',output_example)
        print('-'*40)
        print('-'*40)
        print()
        t += 1
    if t==1:
        break


# ### The model embeds the text in a 300-dimensional space and passes it through two LSTM layers and one GRU layer. The cell state and hidden state are kept from one LSTM to the next. To avoid the issue of repitition, instead of always choosing the most likely character, the model learns a multinomial distribution and draws samples from it, so that a single input will not always produce the same output. With the model output being a probability distribution, the loss function is the negative log likelihood.

# In[5]:


vocab_length = len(char_tokenizer.get_vocabulary())
embedding_dim = 300
input_length = sequence_length - 1
layer_size = 800
states = None
tfd = tfp.distributions
tfpl = tfp.layers

inputs = keras.Input(shape=(None,))
embedding = Embedding(input_dim=vocab_length,
                     output_dim=embedding_dim,
                     input_length=input_length)(inputs)
X = keras.layers.BatchNormalization()(embedding)

lstm = LSTM(layer_size,
            activation='relu',
            kernel_initializer='he_normal',
           return_sequences=True,
           return_state=True)

if states is None:
    states = lstm.get_initial_state(X)

X,hidden_state,cell_state = lstm(embedding,initial_state=states)
X = keras.layers.BatchNormalization()(X)

states = [hidden_state,
          cell_state]
X,hidden_state,cell_state = LSTM(layer_size,
                                 return_sequences=True,
                                 activation='relu',
                                 kernel_initializer='he_normal',
                                 return_state=True)(X,initial_state=states)
X = keras.layers.BatchNormalization()(X)
states = [hidden_state,
          cell_state]
X,cell_state = GRU(layer_size,
                  return_sequences=True,
                  return_state=True)(X,initial_state=states[1])
X = keras.layers.BatchNormalization()(X)
states = [hidden_state,
         cell_state]
X = Dense(tfpl.OneHotCategorical.params_size(vocab_length))(X)
outputs = tfpl.OneHotCategorical(event_size=vocab_length)(X)

sequence_model = keras.Model(inputs=inputs,
                            outputs=outputs)

loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=1e-4)

nll = lambda y_true,y_pred: -y_pred.log_prob(tf.one_hot(y_true,depth=vocab_length))

sequence_model.compile(loss=nll,
                      optimizer=optimizer,
                      metrics=['accuracy'])

sequence_model.summary()


# ### Instead of training a set dataset on a number of epochs, the dataset is first created with a small shift size that increases after one epoch. This allows for slower, more thorough training in the beginning and quicker training towards the end. It also ensures that the data keeps changing, making the training mroe flexible.

# In[6]:


shift_sizes = np.arange(15,40)

for shift in shift_sizes:
    steps_per_epoch = int(len(full_text)/(batch_size*shift))
    dataset = make_dataset(tokenized_text,shift=shift)
    sequence_model.fit(dataset,
                      steps_per_epoch=steps_per_epoch,
                      epochs=1)


# In[7]:


def generate_letter(seed,model):
    seed = seed.replace('\n','')
    split_seed = tf.strings.unicode_split(seed,'UTF-8')
    tokenized_seed = char_tokenizer(split_seed)
    expanded = tf.expand_dims(tokenized_seed,axis=0)
    predictions = model.predict(expanded).squeeze()
    tokens = np.argmax(predictions,axis=-1)
    predicted_str = tf.strings.reduce_join(detokenizer(tokens),axis=-1).numpy().decode()
    predicted_letter = predicted_str[-1]
    return predicted_letter

def generate(letters,seed,model=sequence_model):
    for i in range(letters):
        seed += generate_letter(seed,model)
    return seed


# ### The examples below are created from "seeds" of text that are fed into the neural network to output more characters.

# In[8]:


seed = """He glared at Voldemort, """
generated_text = generate(500,seed)
print(generated_text)
print('-'*50)
print()

seed = """Harry, Ron and Hermione ran up the stairs, trying not to drop the """
generated_text = generate(1200,seed)
print(generated_text)
print('-'*50)
print()

seed = """fought her way across to the stand where Snape stood, and was now racing along the 
row behind him; she didn’t even stop to say sorry as she knocked Professor Quirrell headfirst 
into the row in front. Reaching Snape, she crouched down, pulled out her """
generated_text = generate(1200,seed)
print(generated_text)
print('-'*50)
print()

seed = """Harry gripped his wand tightly, wondering which spell would come in most useful. """
generated_text = generate(1200,seed)
print(generated_text)
print('-'*50)
print()

seed = """Harry looked down the list and found that he was expected in Professor McGonagall’s 
office at half-past two on Monday, which would mean missing most of Divination. He and the other 
fifth years spent a considerable part of the final weekend of the Easter break reading all the 
career information that had been left there for their perusal. “Well, I don’t fancy Healing,” 
said Ron on the last evening of the holidays. He was immersed in a leaflet """
print(generate(1200,seed))
print('-'*50)
print()

seed = """Fourteen times he made me buff up that Quidditch Cup before he was satisfied. And then 
I had another slug attack all over a Special Award for Services to the """
print(generate(1200,seed))
print('-'*50)
print()

seed = """Harry looked down at his History of Magic essay, his quill hanging aimlessly from his hand. 
He couldn't think of how to fill twelve inches of parchment with accounts of Goblin wars. He looked around the common
room where a few fifth years remained huddled over their O.W.L notes near the dying fire."""
print(generate(1200,seed))
print('-'*50)
print()

