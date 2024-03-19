# import streamlit as st
# import pickle
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Ensure you have the necessary NLTK downloads
# nltk.download('punkt')
# nltk.download('stopwords')

# # Initialize PorterStemmer
# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = word_tokenize(text)
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#     text = y[:]
#     y.clear()
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#     text = y[:]
#     y.clear()
#     for i in text:
#         y.append(ps.stem(i))
#     return " ".join(y)

# # Load the vectorizer and models
# tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))
# decision_tree_model = pickle.load(open('model/logisticregression.pkl', 'rb'))  # Correct filename if needed
# random_forest_model = pickle.load(open('model/random_forest.pkl', 'rb'))
# naive_bayes_model = pickle.load(open('model/nayesbayes.pkl', 'rb'))  # Correct filename if needed

# models = {
#     'Decision Tree': decision_tree_model,
#     'Random Forest': random_forest_model,
#     'Naive Bayes': naive_bayes_model,
# }

# st.title("Fake News Detection")

# input_text = st.text_area("Enter the news text")

# if st.button('Predict'):
#     # Preprocess
#     transformed_text = transform_text(input_text)
#     # Vectorize
#     vector_input = tfidf.transform([transformed_text])
#     # Predict and display
#     for model_name, model in models.items():
#         prediction = model.predict(vector_input)[0]
#         result_text = "Real News" if prediction == 1 else "Fake News"
#         st.write(f"{model_name}: {result_text}")

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure you have the necessary NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load the vectorizer and models
tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))
decision_tree_model = pickle.load(open('model/logisticregression.pkl', 'rb'))  # Correct filename if needed
random_forest_model = pickle.load(open('model/random_forest.pkl', 'rb'))
naive_bayes_model = pickle.load(open('model/nayesbayes.pkl', 'rb'))  # Correct filename if needed

models = {
    'Decision Tree': decision_tree_model,
    'Random Forest': random_forest_model,
    'Naive Bayes': naive_bayes_model,
}

# Styling
st.markdown("""
<style>
.title {
    text-align: center;
    color: #1E88E5;
    font-size: 32px;
    margin-bottom: 30px;
    text-shadow: 2px 2px #E3F2FD;
}

.text-input {
    width: 80%;
    height: 150px;
    margin: 0 auto;
    display: block;
    border-radius: 8px;
    border: 2px solid #1E88E5;
    padding: 10px;
    font-size: 16px;
}

.predict-btn {
    background-color: #1E88E5;
    color: white;
    font-size: 18px;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.predict-btn:hover {
    background-color: #0D47A1;
}

.result {
    margin-top: 20px;
    font-size: 30px;
}

.quote {
    margin-top: 40px;
    font-size: 18px;
    color: #7575675;
}

body {
    background-color: #f0f2f5;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: left; color: #1E88E5;'>Farzi</h1>", unsafe_allow_html=True)

st.title("Fake News Detection")

container = st.container()
with container:
    input_text = st.text_area("Enter the news text", height=150)

if st.button('Predict'):
    # Preprocess
    transformed_text = transform_text(input_text)
    # Vectorize
    vector_input = tfidf.transform([transformed_text])
    # Predict and display
    for model_name, model in models.items():
        prediction = model.predict(vector_input)[0]
        result_text = "Real News" if prediction == 1 else "Fake News"
        st.markdown(f"<div class='result'>{model_name}: {result_text}</div>", unsafe_allow_html=True)

# Quote about fake news
quote = """
"Fake news is cheap to produce. Genuine journalism is expensive."

"""
st.markdown(f"<div class='quote'>{quote}</div>", unsafe_allow_html=True)
# import streamlit as st
# import pickle
# import string
# import nltk
# import numpy as np
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.tokenize import word_tokenize
# import gensim.downloader as api
# from tensorflow.keras.models import load_model

# # Ensure you have the necessary NLTK downloads
# nltk.download('punkt')
# nltk.download('stopwords')

# # Initialize PorterStemmer
# ps = PorterStemmer()

# # Load the Word2Vec model
# wv = api.load('word2vec-google-news-300')

# def transform_text(text):
#     text = text.lower()
#     text = word_tokenize(text)
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#     text = y[:]
#     y.clear()
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#     text = y[:]
#     y.clear()
#     for i in text:
#         y.append(ps.stem(i))
#     return " ".join(y)

# def text_to_word2vec_vector(text, wv_model):
#     words = [word for word in text.split() if word in wv_model.key_to_index]
#     if len(words) == 0:
#         return np.zeros(wv_model.vector_size)
#     text_vector = np.mean([wv_model[word] for word in words], axis=0)
#     return text_vector

# # Load the saved machine learning models
# decision_tree_model = pickle.load(open('model/logisticregression.pkl', 'rb'))
# random_forest_model = pickle.load(open('model/random_forest.pkl', 'rb'))
# # naive_bayes_model = pickle.load(open('model/nayesbayes.pkl', 'rb'))
# mlp_model = load_model('model/mlp_model.h5')
# lstm_model = load_model('model/lstm_model.h5')

# # Styling
# st.markdown("""
# <style>
# .title {
#     text-align: center;
#     color: #1E88E5;
#     font-size: 32px;
#     margin-bottom: 30px;
#     text-shadow: 2px 2px #E3F2FD;
# }

# .text-input {
#     width: 80%;
#     height: 150px;
#     margin: 0 auto;
#     display: block;
#     border-radius: 8px;
#     border: 2px solid #1E88E5;
#     padding: 10px;
#     font-size: 16px;
# }

# .predict-btn {
#     background-color: #1E88E5;
#     color: white;
#     font-size: 18px;
#     border: none;
#     padding: 10px 20px;
#     border-radius: 5px;
#     cursor: pointer;
#     transition: background-color 0.3s ease;
# }

# .predict-btn:hover {
#     background-color: #0D47A1;
# }

# .result {
#     margin-top: 20px;
#     font-size: 30px;
# }

# .quote {
#     margin-top: 40px;
#     font-size: 18px;
#     color: #7575675;
# }

# body {
#     background-color: #f0f2f5;
# }
# </style>
# """, unsafe_allow_html=True)

# st.markdown("<h1 style='text-align: left; color: #1E88E5;'>Farzi</h1>", unsafe_allow_html=True)

# st.title("Fake News Detection")

# container = st.container()
# with container:
#     input_text = st.text_area("Enter the news text", height=150)

# if st.button('Predict'):
#     # Preprocess
#     transformed_text = transform_text(input_text)
#     # Vectorize
#     word2vec_input = np.array([text_to_word2vec_vector(transformed_text, wv)])
#     # Predict and display
#     for model_name, model in zip(['Decision Tree', 'Random Forest','MLP', 'LSTM'],
#                                  [decision_tree_model, random_forest_model, mlp_model, lstm_model]):
#         if model_name in ['MLP', 'LSTM']:
#             prediction = (model.predict(word2vec_input) > 0.5).astype("int32")[0][0]
#         else:
#             prediction = model.predict(word2vec_input)[0]
#         result_text = "Real News" if prediction == 1 else "Fake News"
#         st.markdown(f"<div class='result'>{model_name}: {result_text}</div>", unsafe_allow_html=True)

# # Quote about fake news
# quote = """
# "Fake news is cheap to produce. Genuine journalism is expensive."
# — Dan Rather
# """
# st.markdown(f"<div class='quote'>{quote}</div>", unsafe_allow_html=True)



# import streamlit as st
# import pickle
# import string
# import nltk
# import numpy as np
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.tokenize import word_tokenize
# import gensim.downloader as api
# from tensorflow.keras.models import load_model
# import threading
# from tensorflow.keras.optimizers import Adam

# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM

# # Ensure you have the necessary NLTK downloads
# nltk.download('punkt')
# nltk.download('stopwords')

# # Initialize PorterStemmer
# ps = PorterStemmer()

# # Load the Word2Vec model
# wv = None

# def load_word2vec_model():
#     global wv
#     wv = api.load('word2vec-google-news-300')

# def transform_text(text):
#     text = text.lower()
#     text = word_tokenize(text)
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#     text = y[:]
#     y.clear()
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#     text = y[:]
#     y.clear()
#     for i in text:
#         y.append(ps.stem(i))
#     return " ".join(y)

# def text_to_word2vec_vector(text, wv_model):
#     words = [word for word in text.split() if word in wv_model.key_to_index]
#     if len(words) == 0:
#         return np.zeros(wv_model.vector_size)
#     text_vector = np.mean([wv_model[word] for word in words], axis=0)
#     return text_vector

# # Load the saved machine learning models
# decision_tree_model = None
# random_forest_model = None
# # naive_bayes_model = None
# mlp_model = None
# lstm_model = None

# def load_ml_models():
#     global decision_tree_model, random_forest_model, naive_bayes_model, mlp_model, lstm_model
#     decision_tree_model = pickle.load(open('model/logisticregression.pkl', 'rb'))
#     random_forest_model = pickle.load(open('model/random_forest.pkl', 'rb'))
#     # naive_bayes_model = pickle.load(open('model/nayesbayes.pkl', 'rb'))
    
#     # Load the trained MLP model
#     mlp_model = load_model('model/mlp_model.h5',compile=False)
    
#     # Load the trained LSTM model
#     lstm_model = load_model('model/lstm_model.h5',compile=False)

# # Load models asynchronously
# threading.Thread(target=load_word2vec_model).start()
# threading.Thread(target=load_ml_models).start()

# # Styling
# st.markdown("""
# <style>
# .title {
#     text-align: center;
#     color: #1E88E5;
#     font-size: 32px;
#     margin-bottom: 30px;
#     text-shadow: 2px 2px #E3F2FD;
# }

# .text-input {
#     width: 80%;
#     height: 150px;
#     margin: 0 auto;
#     display: block;
#     border-radius: 8px;
#     border: 2px solid #1E88E5;
#     padding: 10px;
#     font-size: 16px;
# }

# .predict-btn {
#     background-color: #1E88E5;
#     color: white;
#     font-size: 18px;
#     border: none;
#     padding: 10px 20px;
#     border-radius: 5px;
#     cursor: pointer;
#     transition: background-color 0.3s ease;
# }

# .predict-btn:hover {
#     background-color: #0D47A1;
# }

# .result {
#     margin-top: 20px;
#     font-size: 30px;
# }

# .quote {
#     margin-top: 40px;
#     font-size: 18px;
#     color: #7575675;
# }

# body {
#     background-color: #f0f2f5;
# }
# </style>
# """, unsafe_allow_html=True)

# st.markdown("<h1 style='text-align: left; color: #1E88E5;'>Farzi</h1>", unsafe_allow_html=True)

# st.title("Fake News Detection")

# container = st.container()
# with container:
#     input_text = st.text_area("Enter the news text", height=150)

# if st.button('Predict'):
#     # Check if models are loaded
#     if wv is None or decision_tree_model is None or random_forest_model is None or mlp_model is None or lstm_model is None:
#         st.warning("Models are still loading. Please wait and try again.")
#     else:
#         # Preprocess
#         transformed_text = transform_text(input_text)
#         # Vectorize
#         word2vec_input = np.array([text_to_word2vec_vector(transformed_text, wv)])
#         # Predict and display
#         for model_name, model in zip(['Decision Tree', 'Random Forest', 'Naive Bayes', 'MLP', 'LSTM'],
#                                      [decision_tree_model, random_forest_model, mlp_model, lstm_model]):
#             if model_name in ['MLP', 'LSTM']:
#                 # Reshape input for LSTM model
#                 word2vec_input_reshaped = np.reshape(word2vec_input, (word2vec_input.shape[0], word2vec_input.shape[1], 1))
#                 prediction = (model.predict(word2vec_input_reshaped) > 0.5).astype("int32")[0][0]
#             else:
#                 prediction = model.predict(word2vec_input)[0]
#             result_text = "Real News" if prediction == 1 else "Fake News"
#             st.markdown(f"<div class='result'>{model_name}: {result_text}</div>", unsafe_allow_html=True)
