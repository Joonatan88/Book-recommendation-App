from flask import Flask, render_template, request
import pandas as pd
from bookRec import clean_data, train_rec_model, recommend_books, books_on_authors

app = Flask(__name__)

df = pd.read_csv("BooksDataset.csv")
cleaned_df = clean_data(df)
tfidf_matrix, vectorizer, processed_df = train_rec_model(cleaned_df)

@app.route('/')
def home():
    #render homepage
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    #handle book recommendations
    return "hello"


if __name__ == "__main__":
    app.run(debug=True)

