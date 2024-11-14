#s-1:-import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load the imdb dataset word_idnex

word_index = imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

#load the pretrained model
model = load_model('simple_rnn_imdb.h5')

#function to decode reviews
def decode_review(review):
     return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
def pre_process(review):#in preprocess we do padding 
    words = review.split()
    encoded_review = [word_index.get(word,2) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], value=0, padding='post', maxlen=250)
    return padded_review


#design the streamlit app
import streamlit as st
st.title('IMDB movie Sentiment Analysis')
st.write("Enter a movie review")
#userinput 
user_input= st.text_area("Movie Review")

if st.button("Classify"):
    pre_processed=pre_process(user_input)
    ##make prediction
    prediction=model.predict(pre_processed)
    sentiment="Positive" if prediction[0][0]> 0.5 else "Negative"

    #Display the result
    st.write(f"sentiment:-{sentiment}")
    st.write(f"Prediction score:-{prediction[0][0]}")
else:
    st.write("please enter a movie review")
