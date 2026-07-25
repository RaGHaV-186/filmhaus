import streamlit as st
import pickle
import pandas as pd
import numpy as np

@st.cache_resource
def load_models():
    with open("models.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_models()

user_item = bundle["user_item"]
similarity = bundle["similarity"]
item_similarity = bundle["item_similarity"]
item_ids = bundle["item_ids"]
popularity = bundle["popularity"]
movies = bundle["movies"]
user_cluster = bundle["user_cluster"]
cluster_top_films = bundle["cluster_top_films"]

def recommend_popular(user_id, k=10):
    seen = set(user_item.loc[user_id][user_item.loc[user_id] > 0].index)
    recs = [film for film in popularity.index if film not in seen]
    return recs[:k]


def recommend_by_history(user_id, k=10):
    user_ratings = user_item.loc[user_id]
    liked_films = user_ratings[user_ratings >= 4].index
    liked_positions = [item_ids.get_loc(mid) for mid in liked_films]
    scores = item_similarity[liked_positions].sum(axis=0)
    scores = pd.Series(scores, index=item_ids)
    already_seen = user_ratings > 0
    scores[already_seen] = 0
    return scores.sort_values(ascending=False).head(k).index.tolist()


def recommend_for_user(user_id, k=10):
    position = user_item.index.get_loc(user_id)
    user_row = similarity[position]
    similar_positions = np.argsort(user_row)[::-1]
    neighbour_positions = similar_positions[1:21]
    neighbour_ratings = user_item.iloc[neighbour_positions]
    film_scores = neighbour_ratings.sum(axis=0)
    already_seen = user_item.loc[user_id] > 0
    film_scores[already_seen] = 0
    return film_scores.sort_values(ascending=False).head(k).index.tolist()


def recommend_by_persona(user_id, k=10):
    cluster = user_cluster[user_id]
    ranked = cluster_top_films[cluster]
    seen = set(user_item.loc[user_id][user_item.loc[user_id] > 0].index)
    recs = [film for film in ranked if film not in seen]
    return recs[:k]

def ids_to_titles(ids):
    return movies[movies["movie_id"].isin(ids)][["title", "genres"]]

st.title("🎬 Filmhaus Recommender")
st.write("Enter a user ID to see recommendations from four different models.")

user_id_input = st.text_input("User ID", value="635")

if user_id_input:
    user_id = int(user_id_input)

    if user_id not in user_item.index:
        st.error(f"User {user_id} has no training history — try another ID.")
    else:
        st.header("⭐ Popular")
        st.table(ids_to_titles(recommend_popular(user_id)))

        st.header("📖 Based on your history")
        st.table(ids_to_titles(recommend_by_history(user_id)))

        st.header("👥 People like you")
        st.table(ids_to_titles(recommend_for_user(user_id)))

        st.header("🎯 Your taste tribe")
        st.table(ids_to_titles(recommend_by_persona(user_id)))