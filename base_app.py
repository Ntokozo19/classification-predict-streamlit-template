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

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("EDSA Team TS4 Climate change Tweet Classifier")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Introduction", "Data Collection","Data Transformation Process", "Prediction"]
	selection = st.sidebar.selectbox("Navigate", options)

	# Building out the "Introduction" page
	if selection == "Introduction":
		st.info("Background")
		# You can read a markdown file from supporting resources folder
		st.markdown("Many companies are built around lessening one’s environmental impact or carbon footprint. "
					"They offer products and services that are environmentally friendly and sustainable, "
					"in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. "
					"This would add to their market research efforts in gauging how their product/service may be received."
					"With this context, We will present to you, a Machine Learning model "
					"that is able to classify whether or not a person believes in climate change, based on their novel tweet data."
					"Providing an accurate and robust solution to this task would give your company access to a broad base of consumer sentiment, "
					"spanning multiple demographic and geographic categories - thus increasing your insights and future marketing strategies.")

		st.subheader("Some links on climate change")
		st.markdown('What is climate change : link')

	# Building out the "Introduction" page
	if selection == "Data Collection":
		st.info("Where does our data come from?")
		# You can read a markdown file from supporting resources folder
		st.markdown("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, "
						"University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between "
						"Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were collected. Each tweet is labelled as one of the following classes:")
		st.markdown("1: The tweet supports the belief of man-made climate change")
		st.markdown("-1: The tweet does not believe in man-made climate change")
		st.markdown("0: The tweet neither supports nor refutes the belief of man-made climate change")
		st.markdown("2: The tweet links to factual news about climate change")




	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with Logistic Regression Model")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter The tweet here","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("This tweet falls under group {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
