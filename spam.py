import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
import pickle
import time
from googletrans import Translator


model = load_model("model.h5")
translator = Translator()

max_vocab_size = 8000
max_sequence_length = 120

def classify_text(text):
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequence)[0][0]

    label = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return label, confidence

def show_page():
    st.write("<h3 style='text-align: center; color: blue;'>شناسایی ایمیل جعلی / اسپم ✉️</h3>", unsafe_allow_html=True)
    st.write("<h6 style='text-align: center; color: black;'>Robo-Ai.ir طراحی و توسعه</h6>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")

    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>تشخیص ایمیل های مشکوک به کلاهبرداری مالی 📮</h6>", unsafe_allow_html=True)
    st.write('')

    with st.sidebar:
        st.write("<h5 style='text-align: center; color: blcak;'>تشخیص ایمیل اسپم با هوش مصنوعی</h5>", unsafe_allow_html=True)
        st.divider()
        st.write("<h5 style='text-align: center; color: black;'>طراحی و توسعه</h5>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: center; color: gray;'>حمیدرضا بهرامی</h5>", unsafe_allow_html=True)

    new_text = st.text_area('متن خود را وارد کنید',height=None,max_chars=None,key=None)
    
    if st.button('ارزیابی'):
        if new_text == "":
            with st.chat_message("assistant"):
                with st.spinner('''درحال ارزیابی'''):
                    time.sleep(1)
                    st.success(u'\u2713''ارزیابی انجام شد')
                    text0 = 'لطفا متن خود را وارد کنید'
                    def stream_data0():
                            for word in text0.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                    st.write_stream(stream_data0)
        
        else:
            out = translator.translate(new_text)
            label, confidence = classify_text(out.text)
            if label == 'Negative':
                with st.chat_message("assistant"):
                    with st.spinner('''درحال ارزیابی'''):
                        time.sleep(1)
                        st.success(u'\u2713''ارزیابی انجام شد')
                        text1 = 'ایمیل / پیام اسپم شناسایی نشد'
                        text2 = 'اطمینان من از دقت محاسبه'
                        text3 = (confidence)
                        text4 = 'Spam email / message is Not identified'
                        text5 = 'My calculated probability'
                        text6 = (confidence)
                        def stream_data1():
                            for word in text1.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data1)
                        def stream_data2():
                            for word in text2.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data2)
                        st.markdown(text3)
                        def stream_data4():
                            for word in text4.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data4)
                        def stream_data5():
                            for word in text5.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data5)
                        st.markdown(text6)
                        

            elif label == 'Positive':
                with st.chat_message("assistant"):
                    with st.spinner('''درحال ارزیابی'''):
                        time.sleep(1)
                        st.success(u'\u2713''ارزیابی انجام شد')
                        text7 = 'ایمیل / پیام اسپم شناسایی شد'
                        text8 = 'اطمینان من از دقت محاسبه'
                        text9 = (confidence)
                        text10 = 'Spam email / message is identified'
                        text11 = 'My calculated probability'
                        text12 = (confidence)
                        def stream_data7():
                            for word in text7.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data7)
                        def stream_data8():
                            for word in text8.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data8)
                        st.markdown(text9)
                        def stream_data10():
                            for word in text10.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data10)
                        def stream_data11():
                            for word in text11.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data11)
                        st.markdown(text12)

    else:
        pass
            
show_page()
