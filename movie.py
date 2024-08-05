import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Load the data
column_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("ml-100k/u.data", sep='\t', names=column_names)

# Load movie titles
movie_titles = pd.read_csv("ml-100k/u.item", sep='|', header=None, encoding='latin-1')
movie_titles = movie_titles[[0, 1]]
movie_titles.columns = ['item_id', 'title']

# Merge dataframes
df = pd.merge(df, movie_titles, on="item_id")

# Creating the movie matrix
moviemat = df.pivot_table(index="user_id", columns="title", values="rating")

# Create a dataframe with ratings and number of ratings
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['num_of_ratings'] = df.groupby('title').count()['rating']
ratings = ratings.sort_values(by='rating', ascending=False)

# Predict function
def predict_movies(movie_name):
    movie_user_ratings = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)

    corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)

    corr_movie = corr_movie.join(ratings['num_of_ratings'])
    predictions = corr_movie[corr_movie['num_of_ratings'] > 100].sort_values('Correlation', ascending=False)

    return predictions

# Prompt user for a movie name
movie_name = input("Enter a movie name: ")
# example: Star Wars (1977)

if movie_name in moviemat.columns:
    predictions = predict_movies(movie_name)
    print(f"Movies similar to '{movie_name}':\n", predictions.head())
else:
    print(f"Movie '{movie_name}' not found in the dataset.")
