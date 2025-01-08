import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Clean dataset
def clean_data(df):
    try:
        print("Data cleaning...")

        if 'Description' not in df.columns:
            raise KeyError("The dataframe does not have a 'Description' column.")

        cleaned_df = df.dropna(subset=['Description'])  # Drop NA values
        cleaned_df = cleaned_df[cleaned_df['Description'].str.strip() != '']  # Drop rows with no content in 'Description'
        cleaned_df['Description'] = cleaned_df['Description'].str.lower()  # Convert to lowercase

        return cleaned_df
   
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return pd.DataFrame() 
    
#Train the model
def train_rec_model(df, description_col = 'Description'):

    print("Model training...")

    vectorizer = TfidfVectorizer(stop_words='english') #Creates TF-IDF vectorizer instance
    tfidf_matrix = vectorizer.fit_transform(df[description_col]) #converts description column from data to numerical vector that are combined to matrix

    return tfidf_matrix, vectorizer, df

#Recommendation on keywords
def recommend_books(keywords, tfidf_matrix, vectorizer, df, number):

    print("Finding similar books...")

    user_vector = vectorizer.transform([keywords]) #user input keywords to vectors

    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten() #compare user input vectors to TF-IDF matrix from database

    top_indices = similarity_scores.argsort()[-number:][::-1] #sort most similar in order

    return df.iloc[top_indices]

#Search books from authors
def books_on_authors(df, author_name):
    result = df[df['Authors'].str.contains(author_name, case=False, na=False)]
    return result

