import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GRU, Input, Dense, TimeDistributed
from keras.models import Model, Sequential
from keras.layers import Activation
from keras.losses import sparse_categorical_crossentropy
import tensorflow


def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x) 
    """
    x_tk = Tokenizer(char_level=True)
    # because input is text, not sequence (list of integer tokens)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # Find the length of the longest string in the dataset.
    if length is None:
        length = max([len(sentence) for sentence in x])
    # Then, pass it to pad_sentences as the maxlen parameter 
    return pad_sequences(x, maxlen=length, padding="post", truncating="post",)

def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk

def simple_model(input_shape, output_sequence_length, code_vocab_size, plaintext_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param code_vocab_size: Number of unique code characters in the dataset
    :param plaintext_vocab_size: Number of unique plaintext characters in the dataset
    :return: Keras model built, but not trained
    """    
    x = Input(shape=input_shape[1:])   # shape(none,54,1) ie   
    # output must be batchsize x timesteps x units
    seq = GRU(units= 64, return_sequences = True, activation="tanh", name='Layer1')(x)
    output = TimeDistributed(Dense(units = plaintext_vocab_size, activation='softmax', name='Layer2'))(seq)
    model = Model(inputs = x, outputs = output)   
    model.compile(optimizer='adam', loss= sparse_categorical_crossentropy, metrics=['accuracy'])  
    model.summary()
    return model

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


codes = load_data('cipher.txt')
plaintext = load_data('plaintext.txt')
preproc_code_sentences, preproc_plaintext_sentences, code_tokenizer, plaintext_tokenizer = preprocess(codes, plaintext)

# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_code_sentences, preproc_plaintext_sentences.shape[1]) # pad code sequences with maxlen 54: shape=10001x54
tmp_x = tmp_x.reshape((-1, preproc_plaintext_sentences.shape[-2], 1))     # reshape padded code seq in 10001 x 54 x 1
simple_rnn_model = simple_model(tmp_x.shape,preproc_plaintext_sentences.shape[1],len(code_tokenizer.word_index)+1,len(plaintext_tokenizer.word_index)+1)
simple_rnn_model.fit(tmp_x, preproc_plaintext_sentences, batch_size=32, epochs=4, validation_split=0.2)
print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], plaintext_tokenizer))

