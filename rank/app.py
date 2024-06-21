import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

app = Flask(__name__ template_folder='template')

# Load and preprocess the data
df = pd.read_csv('ratings_Electronics.csv', header=None)
df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']
df = df.drop('timestamp', axis=1)

# Ensure 'rating' column is numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Drop rows with invalid ratings
df = df.dropna(subset=['rating'])

df_copy = df.copy(deep=True)

# EDA
rows, columns = df.shape
datatypes = df.dtypes.to_dict()
missing_values = df.isna().sum().to_dict()
rating_summary = df['rating'].describe().to_dict()
rating_distribution = df['rating'].value_counts(1).to_dict()
unique_users = df['user_id'].nunique()
unique_items = df['prod_id'].nunique()
most_rated = df.groupby('user_id').size().sort_values(ascending=False)[:10].to_dict()

counts = df['user_id'].value_counts()
df_final = df[df['user_id'].isin(counts[counts >= 50].index)]

final_ratings_matrix = df_final.pivot(index='user_id', columns='prod_id', values='rating').fillna(0)
print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)

given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
density = (given_num_of_ratings / possible_num_of_ratings) * 100
print(f'density: {density:.2f}%')

average_rating = df_final.groupby('prod_id')['rating'].mean()
count_rating = df_final.groupby('prod_id')['rating'].count()
final_rating = pd.DataFrame({'avg_rating': average_rating, 'rating_count': count_rating}).sort_values(by='avg_rating', ascending=False)

def top_n_products(final_rating, n, min_interaction):
    recommendations = final_rating[final_rating['rating_count'] > min_interaction]
    recommendations = recommendations.sort_values('avg_rating', ascending=False)
    return recommendations.index[:n]

def similar_users(user_index, interactions_matrix):
    similarity = []
    for user in range(interactions_matrix.shape[0]):
        sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])
        similarity.append((user, sim))
    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity]
    similarity_score = [tup[1] for tup in similarity]
    most_similar_users.remove(user_index)
    similarity_score.remove(similarity_score[0])
    return most_similar_users, similarity_score

def recommendations(user_index, num_of_products, interactions_matrix):
    most_similar_users = similar_users(user_index, interactions_matrix)[0]
    prod_ids = set(interactions_matrix.columns[np.where(interactions_matrix.loc[user_index] > 0)])
    recommendations = []
    observed_interactions = prod_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_products:
            similar_user_prod_ids = set(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)])
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break
    return recommendations[:num_of_products]

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_index = int(request.form['user_index'])
    num_of_products = int(request.form['num_of_products'])

    try:
        recs = recommendations(user_index, num_of_products, final_ratings_matrix)
        return render_template('index.html', recommendations=recs)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/eda')
def eda():
    eda_info = {
        'rows': rows,
        'columns': columns,
        'datatypes': datatypes,
        'missing_values': missing_values,
        'rating_summary': rating_summary,
        'unique_users': unique_users,
        'unique_items': unique_items,
        'most_rated': most_rated
    }
    return render_template('eda.html', eda_info=eda_info)

if __name__ == '__main__':
    app.run(debug=True)


'''import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

app = Flask(__name__, template_folder='template')

# Load and preprocess the data
df = pd.read_csv('ratings_Electronics.csv', header=None)
df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']
df = df.drop('timestamp', axis=1)
df_copy = df.copy(deep=True)

# EDA
rows, columns = df.shape
datatypes = df.info()
missing_values = df.isna().sum()
rating_summary = df['rating'].describe()
rating_distribution = df['rating'].value_counts(1)
unique_users = df['user_id'].nunique()
unique_items = df['prod_id'].nunique()
most_rated = df.groupby('user_id').size().sort_values(ascending=False)[:10]

counts = df['user_id'].value_counts()
df_final = df[df['user_id'].isin(counts[counts >= 50].index)]

final_ratings_matrix = df_final.pivot(index='user_id', columns='prod_id', values='rating').fillna(0)
print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)

given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
density = (given_num_of_ratings / possible_num_of_ratings) * 100
print(f'density: {density:.2f}%')

average_rating = df_final.groupby('prod_id').mean()['rating']
count_rating = df_final.groupby('prod_id').count()['rating']
final_rating = pd.DataFrame({'avg_rating': average_rating, 'rating_count': count_rating}).sort_values(by='avg_rating', ascending=False)

def top_n_products(final_rating, n, min_interaction):
    recommendations = final_rating[final_rating['rating_count'] > min_interaction]
    recommendations = recommendations.sort_values('avg_rating', ascending=False)
    return recommendations.index[:n]

def similar_users(user_index, interactions_matrix):
    similarity = []
    for user in range(interactions_matrix.shape[0]):
        sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])
        similarity.append((user, sim))
    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity]
    similarity_score = [tup[1] for tup in similarity]
    most_similar_users.remove(user_index)
    similarity_score.remove(similarity_score[0])
    return most_similar_users, similarity_score

def recommendations(user_index, num_of_products, interactions_matrix):
    most_similar_users = similar_users(user_index, interactions_matrix)[0]
    prod_ids = set(interactions_matrix.columns[np.where(interactions_matrix.loc[user_index] > 0)])
    recommendations = []
    observed_interactions = prod_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_products:
            similar_user_prod_ids = set(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)])
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break
    return recommendations[:num_of_products]



import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='template')

# Load and preprocess the data
df = pd.read_csv('ratings_Electronics.csv', header=None)
df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']
df = df.drop('timestamp', axis=1)
df_copy = df.copy(deep=True)

# EDA
rows, columns = df.shape
missing_values = df.isna().sum()
rating_summary = df['rating'].describe()

plt.figure(figsize=(12, 6))
rating_distribution = df['rating'].value_counts(1)
plt.bar(rating_distribution.index, rating_distribution.values)
plt.xlabel('Rating')
plt.ylabel('Proportion')
plt.title('Rating Distribution')
plt.savefig('static/rating_distribution.png')

num_users = df['user_id'].nunique()
num_products = df['prod_id'].nunique()
most_rated = df.groupby('user_id').size().sort_values(ascending=False)[:10]

# Preprocessing
counts = df['user_id'].value_counts()
df_final = df[df['user_id'].isin(counts[counts >= 50].index)]

final_ratings_matrix = df_final.pivot(index='user_id', columns='prod_id', values='rating').fillna(0)

average_rating = df_final.groupby('prod_id').mean()['rating']
count_rating = df_final.groupby('prod_id').count()['rating']
final_rating = pd.DataFrame({'avg_rating': average_rating, 'rating_count': count_rating})
final_rating = final_rating.sort_values(by='avg_rating', ascending=False)

def top_n_products(final_rating, n, min_interaction):
    recommendations = final_rating[final_rating['rating_count'] > min_interaction]
    recommendations = recommendations.sort_values('avg_rating', ascending=False)
    return recommendations.index[:n]

@app.route('/')
def home():
    return render_template('index.html', rows=rows, columns=columns, num_users=num_users, num_products=num_products)

@app.route('/recommend', methods=['POST'])
def recommend():
    num_of_products = int(request.form['num_of_products'])
    min_interaction = int(request.form['min_interaction'])

    try:
        recs = list(top_n_products(final_rating, num_of_products, min_interaction))
        return render_template('index.html', recommendations=recs, rows=rows, columns=columns, num_users=num_users, num_products=num_products)
    except Exception as e:
        return render_template('index.html', error=str(e), rows=rows, columns=columns, num_users=num_users, num_products=num_products)

if __name__ == '__main__':
    app.run(debug=True)'''
