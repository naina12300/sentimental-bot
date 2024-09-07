from flask import Flask, request, render_template
import re
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# Sentiment analysis function
def analyze_sentiment(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    sentiment = model.predict(vectorized_text)[0]
    return sentiment

# Define the route for the homepage
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        sentiment = analyze_sentiment(user_input)
        return render_template("index.html", user_input=user_input, sentiment=sentiment)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
