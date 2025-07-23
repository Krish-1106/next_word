import os
from model import NextWordPredictor

def load_data(file_path):
    """Loading text data from file"""
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def main():
    # Loading data
    data_path = os.path.join('data', 'sample_text.txt')
    text = load_data(data_path)
    
    # Initialising model
    predictor = NextWordPredictor(vocab_size=1000, max_sequence_len=5, embedding_dim=50)
    
    # Preprocessing data
    X, y = predictor.preprocess_data(text)
    
    # Building and training model
    predictor.build_model()
    history = predictor.train(X, y, epochs=100, batch_size=32)
    
    # Saving model
    os.makedirs('models', exist_ok=True)
    predictor.save_model(os.path.join('models', 'next_word_model.h5'))
    
    # Testing prediction
    seed_text = "The quick brown"
    predicted_text = predictor.predict_next_word(seed_text, num_words=3)
    print(f"Seed text: '{seed_text}'")
    print(f"Predicted text: '{predicted_text}'")

if __name__ == "__main__":
    main()