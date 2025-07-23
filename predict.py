import os
from model import NextWordPredictor

def load_data(file_path):
    """Loading text data from file"""
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def main():
    # Loading data for tokenizer
    data_path = os.path.join('data', 'sample_text.txt')
    text = load_data(data_path)
    
    # Initialising model
    predictor = NextWordPredictor(vocab_size=1000, max_sequence_len=5, embedding_dim=50)
    
    # Fitting tokenizer on data
    predictor.tokenizer.fit_on_texts([text])
    
    # Loading trained model
    model_path = os.path.join('models', 'next_word_model.h5')
    predictor.load_model(model_path)
    
    # Interactive predicting
    while True:
        seed_text = input("Enter seed text (or 'q' to quit): ")
        if seed_text.lower() == 'q':
            break
        
        num_words = int(input("How many words to predict? "))
        predicted_text = predictor.predict_next_word(seed_text, num_words=num_words)
        print(f"Predicted text: '{predicted_text}'")

if __name__ == "__main__":
    main()