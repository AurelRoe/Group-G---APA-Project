# emotion_utils.py
import numpy as np
import pandas as pd
from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
word_index = imdb.get_word_index()

# Shift indices by 3 to reserve special tokens
word_index = {word: (index + 3) for word, index in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

import re


max_features = 5000  # Vocabulary size
def encode_text(text, word_index, maxlen=500):
    # Preprocessing (very basic – adjust as needed)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    words = text.split()

    encoded = [1]  # <START>
    for w in words:
        idx = word_index.get(w, 2)          # 2 = <UNK>
        if idx >= max_features:             # “cap” the vocab
            idx = 2                         # map to <UNK>
        encoded.append(idx)

    # pad/truncate to exactly your model’s input_length!
    if len(encoded) < maxlen:
        encoded += [0] * (maxlen - len(encoded))
    else:
        encoded = encoded[:maxlen]

    return encoded


def predict_sent_emotions(texts, model_emotion, model_sentiment, tokenizer,
                          word_index, emotion_columns, max_len=100, threshold=0.5):
    if isinstance(texts, str):
        texts = [texts]

    x_input = np.array([encode_text(text, word_index) for text in texts])
    prediction_sentiment = model_sentiment.predict(x_input)
    predicted_sentiments = ["Positive" if p > 0.5 else "Negative" for p in prediction_sentiment]

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y_pred_proba = model_emotion.predict(padded_sequences)
    y_pred = (y_pred_proba >= threshold).astype(int)

    #if no emotion is predicted, assign the most probable one
    for i in range(len(y_pred)):
        if np.sum(y_pred[i]) == 0:
            y_pred[i, np.argmax(y_pred_proba[i])] = 1

    results = []
    for i in range(len(texts)):
        emotions = [emotion_columns[j] for j, val in enumerate(y_pred[i]) if val == 1]
        results.append([predicted_sentiments[i], emotions])

    return results
