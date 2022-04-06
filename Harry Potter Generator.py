#!/usr/bin/env python
# coding: utf-8

# ### This program creates an algorithm that "learns" Harry Potter in order to be able to generate new text in a similar style.

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import StringLookup,Embedding,LSTM,GRU,Dense,BatchNormalization
from tensorflow.keras import Input
import tensorflow_probability as tfp
import docx
import random
from pathlib import Path
import os

cd = Path.cwd()
filepath = os.path.join(cd,r'OneDrive\Desktop\Datasets\complete_harry_potter.docx')

doc = docx.Document(filepath)

full_text = []

for paragraph in doc.paragraphs:
    text = paragraph.text
    if text.isupper() == False and 'J.K. Rowling' not in text and text != '' and 'Page | ' not in text:
        full_text.append(text)
full_text = '\n'.join(full_text)
full_text = full_text.replace('\n','')


# ### The complete Harry Potter text is tokenized at the character level, meaning that each individual character (letter, number, punctuation, etc.) recieves its own embedding vector.

# In[2]:


unique_characters = sorted(set(full_text))
vocab_size = len(unique_characters)
print('There are {} unique characters. They are: '.format(vocab_size),end='')
print(unique_characters)

char_tokenizer = StringLookup(vocabulary=unique_characters)

detokenizer = StringLookup(vocabulary=char_tokenizer.get_vocabulary(),
                          invert=True)

split_text = tf.strings.unicode_split(full_text,'UTF-8')
tokenized_text = char_tokenizer(split_text)


# ### Once tokenized, the text is divided into sequences of characters (the size of which is defined by the variable sequence_length.) The sequences are then shuffled (with a seed so as to recreate the random shuffle to train on the same train dataset multiple times) and split into training and validation datasets.

# In[3]:


shift = 12
sequence_length = 301
sequences = []
start_point = 0
while True:
    sequence = tokenized_text[start_point:start_point+sequence_length]
    sequences.append(sequence)
    start_point += shift
    if start_point + sequence_length >= len(tokenized_text):
        break
        
sequences = np.array(sequences)
seed = np.random.seed(100)
np.random.shuffle(sequences)

validation_size = int(len(sequences)*.03)
train_sequences = sequences[:-validation_size]
validation_sequences = sequences[-validation_size:]


# ### The training and validation sequences are turned into dataset objects in order to create a pipeline to feed into the model. This process splits each sequence into an input, which is the entire sequence besides the last letter, and an output, which is the entire sequence besides the first letter. 

# In[4]:


batch_size = 128

def make_dataset(sequence_list):
    np.random.shuffle(sequence_list)
    dataset = tf.data.Dataset.from_tensor_slices(sequence_list)
    dataset = dataset.map(lambda sequence: (sequence[:-1],sequence[1:]))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = make_dataset(train_sequences)
validation_dataset = make_dataset(validation_sequences)


# ### Here are some examples of what inputs and outputs look like:

# In[5]:


print('Examples of inputs and outputs:\n')
t = 12
for batch in iter(train_dataset):
    if random.randint(0,250) == 1:
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


# ### The model uses an embedding layer to embed the vocabulary (the list of characters) in a vector space, the dimensionality of which is given by the variable embedding_dim. It then uses two LSTM layers and one GRU layer (with batch normalization in between each one). The recurrent layers output the final hidden state and the cell state, as well as all hidden states for the LSTM layers. 
# ### All too often, text generation gets stuck in a loop, repeating a short sequences of words or characters endlessly. To solve this problem, instead of becoming a deterministic algorithm, the model learns a probability distribution. Characters are then randomly sampled from the distribution. This stochastic approch offers flexibility and randomness that enables the algorithm to avoid getting stuck in any loops. Another advantage of learning a probability distribution is that the model can use the negative log likelihood, which measures how likely the true output is given the model's weights, as a loss function.

# In[6]:


vocab_length = len(char_tokenizer.get_vocabulary())
embedding_dim = 800
input_length = sequence_length - 1
layer_size = 1200
states = None
tfd = tfp.distributions
tfpl = tfp.layers

inputs = keras.Input(shape=(None,))
embedding = Embedding(input_dim=vocab_length,
                     output_dim=embedding_dim,
                     input_length=input_length)(inputs)
X = BatchNormalization()(embedding)

lstm = LSTM(layer_size,
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
                                 return_state=True)(X,initial_state=states)
X = BatchNormalization()(X)
states = [hidden_state,
          cell_state]
X,cell_state = GRU(layer_size,
                  return_sequences=True,
                  return_state=True)(X,initial_state=states[1])
X = BatchNormalization()(X)
states = [hidden_state,
         cell_state]
X = Dense(tfpl.OneHotCategorical.params_size(vocab_length))(X)
outputs = tfpl.OneHotCategorical(event_size=vocab_length)(X)

sequence_model = keras.Model(inputs=inputs,
                            outputs=outputs)

learning_rate = 1e-4
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

nll = lambda y_true,y_pred: -y_pred.log_prob(tf.one_hot(y_true,depth=vocab_length))

sequence_model.compile(loss=nll,
                      optimizer=optimizer,
                      metrics=['accuracy'])

sequence_model.summary()


# ### The model was trained using more than 40 hours of GPU. The weights were then downloaded and uploaded here. A small sample of the validation dataset is used to evlauate the model.

# In[7]:


weights_path = os.path.join(cd,r'OneDrive\Desktop\Datasets\final-weights\harry-potter-weights.h5')
sequence_model.load_weights(weights_path)

sample_size = 512
test_sequences = random.sample(list(validation_sequences),sample_size)
test_sequences = np.array(test_sequences)
test_dataset = make_dataset(test_sequences)
sequence_model.evaluate(test_dataset)


# In[8]:


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

def generate(seed,num_letters=1200,model=sequence_model):
    for i in range(num_letters):
        seed += generate_letter(seed,model)
    return seed


# ### Finally, the model is being tested on a number of "seeds", which are small bits of texts used to start the text generation. Each seed is fed into the above functions and is used to generate more text.

# In[9]:


seed = """He glared at Voldemort, staring into his snake-like eyes. Hooded Death Eaters surriounded them, jeering 
and laughing as their master taunted and tortured Harry. """
print(generate(seed))
print('-'*100+'\n')

seed = """Harry gripped his wand tightly, wondering which spell would come in most useful. """
print(generate(seed))
print('-'*100+'\n')

seed = """Harry was growing more and more frustrated by the assignment Snape had set them. How was he supposed to 
focus when he had a Goblin rebellion, a date, and a summons to the Ministry of Magic to worry about? Absentmindedly
 chewing his quill, he thought of what he might tell Sirius, and what Sirius might think. """
print(generate(seed))
print('-'*100+'\n')

seed = """Harry looked down at his History of Magic essay, his quill hanging aimlessly from his hand. 
He couldn't think of how to fill twelve inches of parchment with accounts of Goblin wars. He looked around the common
room where a few fifth years remained huddled over their O.W.L notes near the dying fire."""
print(generate(seed))
print('-'*100+'\n')

