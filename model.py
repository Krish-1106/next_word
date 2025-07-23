import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class NextWordPredictor:
    def __init__(self, vocab_size=5000, max_sequence_len=10, embedding_dim=100):
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        self.model = None
        
    def preprocess_data(self, text):
        """Preprocessing the text data for training"""
        # Tokenizing the text
        self.tokenizer.fit_on_texts([text])
        
        # Creating sequences
        input_sequences = []
        for line in text.split('.'):
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        
        # Padding sequences
        input_sequences = pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre')
        
        # Creating predictors and target
        X, y = input_sequences[:,:-1], input_sequences[:,-1]
        
        # Converting target to one-hot encoding
        y = tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)
        
        return X, y
    
    def build_model(self):
        """Building the LSTM+GRU model"""
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_len-1))
        model.add(LSTM(128, return_sequences=True))
        model.add(GRU(128))
        model.add(Dropout(0.2))
        model.add(Dense(self.vocab_size, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return model
    
    def train(self, X, y, epochs=50, batch_size=64, validation_split=0.1):
        """Training the model"""
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        return history
    
    def save_model(self, filepath):
        """Saving the model to disk"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Loading a saved model"""
        self.model = tf.keras.models.load_model(filepath)
    
    def predict_next_word(self, text, num_words=1):
        """Predicting the next word given a seed text"""
        for _ in range(num_words):
            token_list = self.tokenizer.texts_to_sequences([text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
            
            predicted_probs = self.model.predict(token_list, verbose=0)[0]
            predicted_index = np.argmax(predicted_probs)
            
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
            
            text += " " + output_word
        
        return text