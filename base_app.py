"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os


# Data dependencies
import pandas as pd
import plotly.express as px
#from plotly.subplots import make_subplots
#import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
import re




# Vectorizer
news_vectorizer = open("resources/gst_model.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

DATA_URL = (
    "resources/train.csv"
)

st.title("Sentiment Analysis of Tweets about climate change")
st.markdown("This App is a dashboard used "
            "to analyze sentiments of tweets about climate change and to make predictions of the sentiment based on a tweet")
st.markdown("Use the side bar to explore the data and make predictions")
st.sidebar.title("Analysis of Tweet Sentiments")


@st.cache(persist=True, allow_output_mutation=True)
def load_data():
    data = pd.read_csv(DATA_URL)

    return data

data = load_data()




#Display # of tweets by sentiment
st.sidebar.markdown("### View the number of tweets by sentiment")
select = st.sidebar.selectbox('Choose the Visualization type', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = data['sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Number of tweets by sentiment")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
        st.markdown("1: The tweet supports the belief of man-made climate change")
        st.markdown("-1: The tweet does not believe in man-made climate change")
        st.markdown("0: The tweet neither supports nor refutes the belief of man-made climate change")
        st.markdown("2: The tweet links to factual news about climate change")
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)
        st.markdown("1: The tweet supports the belief of man-made climate change")
        st.markdown("-1: The tweet does not believe in man-made climate change")
        st.markdown("0: The tweet neither supports nor refutes the belief of man-made climate change")
        st.markdown("2: The tweet links to factual news about climate change")

#Frquent words word cloud
st.sidebar.markdown(" ### Frequent Words used for each sentiment")
word_sentiment = st.sidebar.radio('Which sentiment would you like to view?', ('Pro', 'Neutral', 'Anti', 'News'))
news = data[data['sentiment'] == 2]['message']
pro = data[data['sentiment'] == 1]['message']
neutral =data[data['sentiment'] ==0]['message']
anti = data[data['sentiment'] ==-1]['message']
if not st.sidebar.checkbox("Hide", True, key=1):
    if word_sentiment == 'Pro':
        st.markdown("### Frequent words used in Pro climate change tweets")
        pro = [word for line in pro for word in line.split()]
        pro = WordCloud(
            background_color='white',
            max_words=50,
            max_font_size=100,
            scale=5,
            random_state=1,
            collocations=False,
            normalize_plurals=False
        ).generate(' '.join(pro))
        fig, ax = plt.subplots()
        ax.imshow(pro)
        plt.xticks([])
        plt.yticks([])
        st.pyplot(fig)
        st.markdown("We see words like believe, combat, fight, real and action which represent the pro climate change supporters who believe that climate change is real and that action needs to be taken stop it.")

    if word_sentiment == 'Anti':
        st.markdown("### Frequent words used in Anti climate change tweets")
        anti = [word for line in anti for word in line.split()]
        anti = WordCloud(
            background_color='white',
            max_words=50,
            max_font_size=100,
            scale=5,
            random_state=1,
            collocations=False,
            normalize_plurals=False
        ).generate(' '.join(anti))
        fig, ax = plt.subplots()
        ax.imshow(anti)
        plt.xticks([])
        plt.yticks([])
        st.pyplot(fig)

    if word_sentiment == 'Neutral':
        st.markdown("### Frequent words used in Neutral climate change tweets")
        neutral = [word for line in neutral for word in line.split()]
        neutral = WordCloud(
            background_color='white',
            max_words=50,
            max_font_size=100,
            scale=5,
            random_state=1,
            collocations=False,
            normalize_plurals=False
        ).generate(' '.join(neutral))
        fig, ax = plt.subplots()
        ax.imshow(neutral)
        plt.xticks([])
        plt.yticks([])
        st.pyplot(fig)

    if word_sentiment == 'News':
        st.markdown("### Frequent words used in News climate change tweets")
        news = [word for line in news for word in line.split()]
        news = WordCloud(
            background_color='white',
            max_words=50,
            max_font_size=100,
            scale=5,
            random_state=1,
            collocations=False,
            normalize_plurals=False
        ).generate(' '.join(news))
        fig, ax = plt.subplots()
        ax.imshow(news)
        plt.xticks([])
        plt.yticks([])
        st.pyplot(fig)


#Frequent words word cloud
st.sidebar.markdown(" ### Make Prediction")
if not st.sidebar.checkbox("Hide", True, key=3):
    st.markdown("### Make a prediction of the sentiment of a tweet")
    tweet_text = st.text_area("Enter tweet to get a prediction", "Type Here")
    if st.button("Classify"):
                # Transforming user input with vectorizer
                vect_text = tweet_cv.transform([tweet_text]).toarray()
                # Load your .pkl file with the model of your choice + make predictions
                # Try loading in multiple models to give the user a choice
                predictor = joblib.load(open(os.path.join("resources/gst_model.pkl"),"rb"))
                prediction = predictor.predict(vect_text)

                # When model has successfully run, will print prediction
                # You can use a dictionary or similar structure to make this output
                # more human interpretable.
                st.success("Text Categorized as: {}".format(prediction))



