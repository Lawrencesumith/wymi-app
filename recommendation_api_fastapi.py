import psycopg2
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
from typing import List, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from rapidfuzz import fuzz, process
from pydantic import BaseModel, field_validator, ValidationError
import uuid
from psycopg2.extras import execute_values
import re
import os
from dotenv import load_dotenv

# Load environment variables and log them for debugging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Fixed typo: _name_ to __name__
logger.info(f"DB_NAME: {os.getenv('DB_NAME')}")
logger.info(f"DB_USER: {os.getenv('DB_USER')}")
logger.info(f"DB_HOST: {os.getenv('DB_HOST')}")
logger.info(f"DB_PORT: {os.getenv('DB_PORT')}")
logger.info(f"API_KEY: {os.getenv('API_KEY')}")

# Database configuration
db_params = {
    'dbname': os.getenv('DB_NAME', 'wymi_db_2b3w'),
    'user': os.getenv('DB_USER', 'wymi_db_2b3w_user'),
    'password': os.getenv('DB_PASSWORD', 'NSzjhMW59KmdMMPMzj49bNFBSPkUT4te'),
    'host': os.getenv('DB_HOST', 'dpg-d26as4be5dus73derc50-a.oregon-postgres.render.com'),
    'port': int(os.getenv('DB_PORT', '5432'))  # Cast to int
}

user_factors = None
item_factors = None
products = None
user_price_preferences = None
popular_products = None
user_item_matrix = None
user_profiles = None
global_popularity = None

def matrix_factorization(R, K=30, steps=300, alpha=0.005, beta=0.04):
    num_users, num_items = R.shape
    P = np.random.rand(num_users, K)
    Q = np.random.rand(num_items, K)
    R = R.to_numpy()
    prev_error = float('inf')
    patience = 20
    min_delta = 0.001
    patience_counter = 0
    for step in range(steps):
        for i in range(num_users):
            for j in range(num_items):
                if R[i, j] > 0:
                    eij = R[i, j] - np.dot(P[i, :], Q[j, :])
                    for k in range(K):
                        P[i, k] += alpha * (2 * eij * Q[j, k] - beta * P[i, k])
                        Q[j, k] += alpha * (2 * eij * P[i, k] - beta * Q[j, k])
        error = 0
        for i in range(num_users):
            for j in range(num_items):
                if R[i, j] > 0:
                    error += (R[i, j] - np.dot(P[i, :], Q[j, :])) ** 2
        if step % 20 == 0:
            logger.info(f"Step {step}, error: {error:.4f}")
        if prev_error - error < min_delta:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at step {step}")
                break
        else:
            patience_counter = 0
        prev_error = error
    logger.info(f"Final error: {error:.4f}")
    return P, Q

def compute_global_popularity(products: pd.DataFrame):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT product_id, activity_type, COUNT(*) as count
            FROM user_activity_log
            WHERE activity_type IN ('view', 'click', 'add_to_cart', 'wishlist')
            GROUP BY product_id, activity_type;
            """
        )
        activity_counts = pd.DataFrame(cursor.fetchall(), columns=['product_id', 'activity_type', 'count'])
        cursor.execute(
            """
            SELECT product_id, COUNT(*) as count
            FROM order_item
            GROUP BY product_id;
            """
        )
        order_counts = pd.DataFrame(cursor.fetchall(), columns=['product_id', 'count']).assign(activity_type='order')
        activity_weights = {
            'view': 0.5,
            'click': 0.7,
            'add_to_cart': 1.0,
            'wishlist': 0.9,
            'order': 1.2
        }
        all_counts = pd.concat([activity_counts, order_counts], ignore_index=True)
        all_counts['weighted_count'] = all_counts.apply(lambda x: x['count'] * activity_weights.get(x['activity_type'], 0.5), axis=1)
        popularity = all_counts.groupby('product_id')['weighted_count'].sum().reset_index()
        popularity = popularity.merge(products[['product_id', 'category_name', 'material', 'price']], on='product_id', how='left')
        popularity['score'] = popularity['weighted_count'] / (popularity['weighted_count'].max() or 1)
        popularity['predicted_rating'] = 3.5
        popularity = popularity.sort_values('score', ascending=False)
        logger.info(f"Computed global popularity for {len(popularity)} products")
        return popularity
    except Exception as e:
        logger.error(f"Error computing global popularity: {e}")
        return products[['product_id', 'category_name', 'material', 'price']].copy().assign(score=0.1, predicted_rating=3.0)
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def load_data():
    global user_factors, item_factors, products, user_price_preferences, popular_products, user_item_matrix, user_profiles, global_popularity
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM category;")
        category_count = cursor.fetchone()[0]
        logger.info(f"Found {category_count} categories")
        if category_count == 0:
            logger.error("Category table is empty")
            return False
        cursor.execute("SELECT COUNT(*) FROM products;")
        product_count = cursor.fetchone()[0]
        logger.info(f"Found {product_count} products")
        if product_count == 0:
            logger.error("Products table is empty")
            return False
        cursor.execute("SELECT products.product_id, products.category_id, products.material, products.price, category.category_name FROM products JOIN category ON products.category_id = category.category_id;")
        products_data = cursor.fetchall()
        if not products_data:
            logger.error("No products found in query")
            return False
        products = pd.DataFrame(products_data, columns=['product_id', 'category_id', 'material', 'price', 'category_name'])
        products['price'] = products['price'].astype(float)
        logger.info(f"Loaded {len(products)} products")
        cursor.execute("SELECT user_id, product_id, rating FROM interactions WHERE product_id <= 50 AND rating > 0 AND rating <= 5;")
        interactions = pd.DataFrame(cursor.fetchall(), columns=['user_id', 'product_id', 'rating'])
        cursor.execute("SELECT user_id, product_id FROM order_item WHERE product_id <= 50;")
        order_items = pd.DataFrame(cursor.fetchall(), columns=['user_id', 'product_id']).assign(rating=4.0)
        cursor.execute("SELECT user_id, product_id, activity_type FROM user_activity_log WHERE product_id <= 50 AND activity_type IN ('view', 'click', 'add_to_cart', 'wishlist');")
        activities = pd.DataFrame(cursor.fetchall(), columns=['user_id', 'product_id', 'activity_type'])
        activities['rating'] = activities['activity_type'].map({'view': 3.0, 'click': 3.5, 'add_to_cart': 4.0, 'wishlist': 4.0})
        interactions_df = pd.concat([interactions, order_items, activities[['user_id', 'product_id', 'rating']]], ignore_index=True)
        interactions_df = interactions_df[(interactions_df['rating'] > 0) & (interactions_df['rating'] <= 5)]
        if interactions_df.empty:
            logger.warning("No valid interactions, using fallback popular products")
            user_item_matrix = pd.DataFrame(index=[str(i) for i in range(1000, 10000)], columns=products['product_id']).fillna(0)
            user_factors = pd.DataFrame(0, index=user_item_matrix.index, columns=[f'factor_{i}' for i in range(30)])
            item_factors = pd.DataFrame(0, index=user_item_matrix.columns, columns=[f'factor_{i}' for i in range(30)])
        else:
            user_item_matrix = interactions_df.pivot_table(index='user_id', columns='product_id', values='rating', aggfunc='mean').fillna(0)
            P, Q = matrix_factorization(user_item_matrix)
            user_factors = pd.DataFrame(P, index=user_item_matrix.index, columns=[f'factor_{i}' for i in range(30)])
            item_factors = pd.DataFrame(Q, index=user_item_matrix.columns, columns=[f'factor_{i}' for i in range(30)])
        cursor.execute("SELECT user_id, price FROM order_item WHERE product_id <= 50;")
        order_prices = pd.DataFrame(cursor.fetchall(), columns=['user_id', 'price'])
        order_prices['price'] = order_prices['price'].astype(float)
        cursor.execute("SELECT user_id, product_id FROM user_activity_log WHERE product_id <= 50 AND activity_type IN ('view', 'click', 'add_to_cart', 'wishlist');")
        activity_prices = pd.DataFrame(cursor.fetchall(), columns=['user_id', 'product_id'])
        activity_prices = activity_prices.merge(products[['product_id', 'price']], on='product_id', how='left')
        user_price_prefs = pd.concat([order_prices.groupby('user_id')['price'].mean().reset_index(), 
                                     activity_prices.groupby('user_id')['price'].mean().reset_index()]).groupby('user_id')['price'].mean().reset_index()
        user_price_preferences = user_price_prefs.set_index('user_id')['price']
        cursor.execute("SELECT product_id, COUNT(*) as view_count FROM user_activity_log WHERE product_id <= 50 AND activity_type IN ('view', 'click', 'add_to_cart', 'wishlist') GROUP BY product_id;")
        view_counts = pd.DataFrame(cursor.fetchall(), columns=['product_id', 'view_count'])
        cursor.execute("SELECT product_id, AVG(rating) as avg_rating FROM interactions WHERE product_id <= 50 GROUP BY product_id;")
        avg_ratings = pd.DataFrame(cursor.fetchall(), columns=['product_id', 'avg_rating'])
        avg_ratings['avg_rating'] = avg_ratings['avg_rating'].astype(float)
        popular_products = view_counts.merge(avg_ratings, on='product_id', how='outer').fillna({'view_count': 0, 'avg_rating': 0})
        popular_products['score'] = 0.7 * popular_products['view_count'] / (popular_products['view_count'].max() or 1) + 0.3 * popular_products['avg_rating'] / 5.0
        if popular_products.empty:
            logger.warning("No popular products, using default product scores")
            popular_products = products[['product_id']].copy()
            popular_products['score'] = 0.1
            popular_products['avg_rating'] = 3.0
        else:
            popular_products = popular_products.sort_values('score', ascending=False).head(10)
        cursor.execute("SELECT user_id, age, gender FROM users;")
        user_profiles_data = cursor.fetchall()
        user_profiles = pd.DataFrame(user_profiles_data, columns=['user_id', 'age', 'gender'])
        cursor.execute("SELECT user_id, product_id FROM recommended_products WHERE product_id <= 50;")
        recommended_products_data = pd.DataFrame(cursor.fetchall(), columns=['user_id', 'product_id'])
        global_popularity = compute_global_popularity(products)
        logger.info(f"Loaded {len(products)} products, {len(interactions_df)} interactions, {len(user_profiles)} user profiles, {len(recommended_products_data)} recommended products, {len(global_popularity)} global popularity scores")
        return True
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    if not load_data():
        logger.error("Failed to load data, starting API with limited functionality")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="WYMI Recommendation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to Render URL after deployment
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type"],
)

API_KEY = "wymi-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

class Recommendation(BaseModel):
    product_id: int
    category: str
    material: str
    price: float
    predicted_rating: float

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]
    suggestion: Optional[str] = None

class ProductResponse(BaseModel):
    product_id: int
    category: str
    material: str
    price: float
    user_id: str

class SearchRequest(BaseModel):
    user_id: str
    query: str

class ActivityRequest(BaseModel):
    user_id: str
    product_id: Optional[int] = None
    activity_type: str

    @field_validator('product_id')
    @classmethod
    def handle_null_product_id(cls, v):
        if v is None:
            return None
        if not isinstance(v, int):
            raise ValueError("product_id must be an integer or null")
        return v

class SignupRequest(BaseModel):
    name: str
    age: int
    gender: str
    email: str
    mobile: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    user_id: str

class UserActivityResponse(BaseModel):
    user_id: str
    product_id: Optional[int]
    product_name: Optional[str]
    activity_type: str
    created_at: str

def content_based_recommendations(query: str, products: pd.DataFrame, n: int = 5):
    query = query.lower().strip()
    valid_materials = list(products['material'].str.lower().unique())
    valid_categories = list(products['category_name'].str.lower().unique())
    material_match = process.extractOne(query, valid_materials, scorer=fuzz.partial_ratio, score_cutoff=70)
    category_match = process.extractOne(query, valid_categories, scorer=fuzz.partial_ratio, score_cutoff=70)
    suggestion = None
    corrected_query = query
    is_material_search = material_match and material_match[1] >= 70
    if is_material_search:
        corrected_query = material_match[0]
        if corrected_query != query:
            suggestion = f"Did you mean {corrected_query}?"
            logger.info(f"Query '{query}' corrected to '{corrected_query}'")
        matches = products[products['material'].str.lower() == corrected_query].copy()
    elif category_match and category_match[1] >= 70:
        corrected_query = category_match[0]
        if corrected_query != query:
            suggestion = f"Did you mean {corrected_query}?"
            logger.info(f"Query '{query}' corrected to '{corrected_query}'")
        matches = products[products['category_name'].str.lower() == corrected_query].copy()
    else:
        matches = products[
            products['category_name'].str.lower().str.contains(query, na=False) |
            products['material'].str.lower().str.contains(query, na=False)
        ].copy()
    if len(matches) == 0:
        logger.info(f"No matches for '{corrected_query}', using fallback")
        matches = products.head(n).copy()
        matches['score'] = 0.1
        matches['predicted_rating'] = 3.0
    else:
        matches['score'] = (
            matches['material'].str.lower().str.contains(corrected_query, na=False).astype(int) * (0.8 if is_material_search else 0.4) +
            matches['category_name'].str.lower().str.contains(corrected_query, na=False).astype(int) * (0.2 if is_material_search else 0.6)
        )
        matches['predicted_rating'] = 3.5
    matches['score'] = matches['score'].fillna(0.1)
    matches['predicted_rating'] = matches['predicted_rating'].fillna(3.0)
    matches = matches.sort_values('score', ascending=False).head(n)
    recommendations = [
        Recommendation(
            product_id=int(row['product_id']),
            category=str(row['category_name']),
            material=str(row['material']),
            price=float(row['price']),
            predicted_rating=float(row['predicted_rating'])
        )
        for _, row in matches.iterrows()
    ]
    return recommendations, suggestion

def activity_based_recommendations(user_id: str, products: pd.DataFrame, n: int = 5):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT product_id, activity_type, created_at FROM user_activity_log WHERE user_id = %s AND activity_type IN ('view', 'click', 'add_to_cart', 'wishlist') ORDER BY created_at DESC LIMIT 5;",
            (user_id,)
        )
        activities = pd.DataFrame(cursor.fetchall(), columns=['product_id', 'activity_type', 'created_at'])
        conn.close()
        if activities.empty:
            recs = products.head(n).copy()
            recs['score'] = 0.1
            recs['predicted_rating'] = 3.0
            recommendations = [
                Recommendation(
                    product_id=int(row['product_id']),
                    category=str(row['category_name']),
                    material=str(row['material']),
                    price=float(row['price']),
                    predicted_rating=float(row['predicted_rating'])
                )
            for _, row in recs.iterrows()
            ]
            return recommendations, None
        activities['weight'] = 1.0 / (1 + (pd.Timestamp.now() - pd.to_datetime(activities['created_at'])).dt.total_seconds() / 3600)
        activities['score'] = activities['activity_type'].map({'view': 0.8, 'click': 1.0, 'add_to_cart': 1.2, 'wishlist': 1.2}) * activities['weight']
        activity_products = products[products['product_id'].isin(activities['product_id'])].copy()
        activity_products = activity_products.merge(activities[['product_id', 'score']], on='product_id')
        categories = activity_products['category_name'].str.lower().unique()
        materials = activity_products['material'].str.lower().unique()
        matches = products.copy()
        matches['score'] = 0.0
        for category in categories:
            matches['score'] += matches['category_name'].str.lower().str.contains(category, na=False).astype(int) * 0.5
        for material in materials:
            matches['score'] += matches['material'].str.lower().str.contains(material, na=False).astype(int) * 0.5
        matches = matches.merge(activity_products[['product_id', 'score']].groupby('product_id').sum(), on='product_id', how='left', suffixes=('', '_activity'))
        matches['score'] = matches['score'].fillna(0) + matches['score_activity'].fillna(0) * 1.5
        matches = matches[matches['score'] > 0].sort_values('score', ascending=False).head(n)
        if len(matches) < n:
            matches = pd.concat([matches, products.head(n - len(matches))]).drop_duplicates().head(n)
        matches['predicted_rating'] = 3.5
        matches['score'] = matches['score'].fillna(0.1)
        matches['predicted_rating'] = matches['predicted_rating'].fillna(3.0)
        recommendations = [
            Recommendation(
                product_id=int(row['product_id']),
                category=str(row['category_name']),
                material=str(row['material']),
                price=float(row['price']),
                predicted_rating=float(row['predicted_rating'])
            )
            for _, row in matches.iterrows()
        ]
        return recommendations, None
    except Exception as e:
        logger.error(f"Error in activity recs for {user_id}: {e}")
        recs = products.head(n).copy()
        recs['score'] = 0.1
        recs['predicted_rating'] = 3.0
        recommendations = [
            Recommendation(
                product_id=int(row['product_id']),
                category=str(row['category_name']),
                material=str(row['material']),
                price=float(row['price']),
                predicted_rating=float(row['predicted_rating'])
            )
            for _, row in recs.iterrows()
        ]
        return recommendations, None

def product_based_recommendations(product_id: int, user_id: str, products: pd.DataFrame, user_profiles: pd.DataFrame, n: int = 5):
    try:
        product = products[products['product_id'] == product_id]
        if product.empty:
            logger.error(f"Product {product_id} not found")
            recs = products.head(n).copy()
            recs['score'] = 0.1
            recs['predicted_rating'] = 3.0
            recommendations = [
                Recommendation(
                    product_id=int(row['product_id']),
                    category=str(row['category_name']),
                    material=str(row['material']),
                    price=float(row['price']),
                    predicted_rating=float(row['predicted_rating'])
                )
                for _, row in recs.iterrows()
            ]
            return recommendations, None
        category = product['category_name'].iloc[0].lower()
        material = product['material'].iloc[0].lower()
        matches = products[products['product_id'] != product_id].copy()
        matches['score'] = (
            matches['category_name'].str.lower().str.contains(category, na=False).astype(int) * 0.6 +
            matches['material'].str.lower().str.contains(material, na=False).astype(int) * 0.4
        )
        user_price = user_price_preferences.get(user_id, products['price'].mean())
        price_range_min = user_price * 0.8
        price_range_max = user_price * 1.2
        matches = matches[matches['price'].between(price_range_min, price_range_max)]
        user_profile = user_profiles[user_profiles['user_id'] == user_id]
        if not user_profile.empty:
            gender = user_profile['gender'].iloc[0].lower()
            age = user_profile['age'].iloc[0]
            if gender == 'female':
                matches['score'] += matches['category_name'].str.lower().isin(['rings', 'earrings', 'necklace', 'maang tikka', 'nose pins', 'waist chains']).astype(int) * 0.2
            elif gender == 'male':
                matches['score'] += matches['category_name'].str.lower().isin(['bracelets', 'brooches']).astype(int) * 0.2
            if 13 <= age <= 25:
                matches['score'] += matches['category_name'].str.lower().isin(['bracelets', 'anklets', 'nose pins']).astype(int) * 0.1
            elif 26 <= age <= 40:
                matches['score'] += matches['category_name'].str.lower().isin(['rings', 'earrings', 'necklace']).astype(int) * 0.1
            elif age > 40:
                matches['score'] += matches['category_name'].str.lower().isin(['bangles', 'waist chains', 'brooches']).astype(int) * 0.1
        matches = matches.sort_values('score', ascending=False).head(n)
        if len(matches) < n:
            matches = pd.concat([matches, products[products['product_id'] != product_id].head(n - len(matches))]).drop_duplicates().head(n)
        matches['predicted_rating'] = 3.5
        matches['score'] = matches['score'].fillna(0.1)
        matches['predicted_rating'] = matches['predicted_rating'].fillna(3.0)
        recommendations = [
            Recommendation(
                product_id=int(row['product_id']),
                category=str(row['category_name']),
                material=str(row['material']),
                price=float(row['price']),
                predicted_rating=float(row['predicted_rating'])
            )
            for _, row in matches.iterrows()
        ]
        return recommendations, None
    except Exception as e:
        logger.error(f"Error in product recs for {product_id}: {e}")
        recs = products[products['product_id'] != product_id].head(n).copy()
        recs['score'] = 0.1
        recs['predicted_rating'] = 3.0
        recommendations = [
            Recommendation(
                product_id=int(row['product_id']),
                category=str(row['category_name']),
                material=str(row['material']),
                price=float(row['price']),
                predicted_rating=float(row['predicted_rating'])
            )
            for _, row in recs.iterrows()
        ]
        return recommendations, None

def recommend_products(user_id: str, user_factors, item_factors, products, user_price_preferences, popular_products, user_item_matrix, user_profiles, global_popularity, n: int = 5):
    logger.info(f"Recommendations for user {user_id}")
    try:
        if products is None or products.empty:
            raise HTTPException(status_code=500, detail="No product data")
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        is_anonymous = user_id.startswith('anon_')
        if is_anonymous:
            if global_popularity is None or global_popularity.empty:
                logger.warning("Global popularity data not available")
                global_popularity = compute_global_popularity(products)
            recs = global_popularity.head(n).copy()
            recommendations = [
                Recommendation(
                    product_id=int(row['product_id']),
                    category=str(row['category_name']),
                    material=str(row['material']),
                    price=float(row['price']),
                    predicted_rating=float(row['predicted_rating'])
                )
                for _, row in recs.iterrows()
            ]
            cursor.execute(
                "SELECT product_id, activity_type FROM user_activity_log WHERE user_id = %s AND activity_type IN ('view', 'click', 'add_to_cart', 'wishlist') ORDER BY created_at DESC LIMIT 5;",
                (user_id,)
            )
            anon_activities = pd.DataFrame(cursor.fetchall(), columns=['product_id', 'activity_type'])
            if not anon_activities.empty:
                activity_products = products[products['product_id'].isin(anon_activities['product_id'])].copy()
                activity_scores = anon_activities.groupby('product_id').size().reset_index(name='count')
                activity_scores['score'] = activity_scores['count'] * 0.3
                recs = recs.merge(activity_scores[['product_id', 'score']], on='product_id', how='left')
                recs['score'] = recs['score'].fillna(0) + recs['score_y'].fillna(0)
                recs = recs.sort_values('score', ascending=False).head(n)
                recommendations = [
                    Recommendation(
                        product_id=int(row['product_id']),
                        category=str(row['category_name']),
                        material=str(row['material']),
                        price=float(row['price']),
                        predicted_rating=float(row['predicted_rating'])
                    )
                    for _, row in recs.iterrows()
                ]
            execute_values(
                cursor,
                "INSERT INTO recommended_products (id, user_id, product_id) VALUES %s;",
                [(str(uuid.uuid4()), user_id, r.product_id) for r in recommendations]
            )
            conn.commit()
            return recommendations, "Popular products based on global user activity"
        cursor.execute("SELECT user_id FROM users WHERE user_id = %s;", (user_id,))
        if not cursor.fetchone():
            logger.warning(f"User {user_id} not found, returning popular products")
            recs = popular_products.head(n).merge(products[['product_id', 'category_name', 'material', 'price']], on='product_id', how='left').copy()
            recs['predicted_rating'] = recs['avg_rating'].fillna(3.0)
            recommendations = [
                Recommendation(
                    product_id=int(row['product_id']),
                    category=str(row['category_name']),
                    material=str(row['material']),
                    price=float(row['price']),
                    predicted_rating=float(row['predicted_rating'])
                )
                for _, row in recs.iterrows()
            ]
            execute_values(
                cursor,
                "INSERT INTO recommended_products (id, user_id, product_id) VALUES %s;",
                [(str(uuid.uuid4()), user_id, r.product_id) for r in recommendations]
            )
            conn.commit()
            return recommendations, "Popular products recommended due to missing user data"
        cursor.execute("SELECT COUNT(*) FROM interactions WHERE user_id = %s;", (user_id,))
        interaction_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM order_item WHERE user_id = %s;", (user_id,))
        order_count = cursor.fetchone()[0]
        is_new_user = interaction_count == 0 and order_count == 0
        user_price = user_price_preferences.get(user_id, products['price'].mean())
        price_range_min = user_price * 0.8
        price_range_max = user_price * 1.2
        recs = products[['product_id', 'category_name', 'material', 'price']].copy()
        recs['score'] = 0.0
        recs['predicted_rating'] = 3.0
        activity_weight = 0.6 if is_new_user else 0.4
        activity_recs, _ = activity_based_recommendations(user_id, products, n=10)
        activity_scores = {r.product_id: r.predicted_rating * activity_weight for r in activity_recs}
        recs['score'] += recs['product_id'].map(activity_scores).fillna(0)
        if not is_new_user and user_id in user_factors.index:
            predicted_ratings = np.dot(user_factors.loc[user_id], item_factors.T)
            collab_df = pd.DataFrame({'product_id': item_factors.index, 'collab_score': predicted_ratings})
            collab_df['collab_score'] = (collab_df['collab_score'] / collab_df['collab_score'].max()) * 0.2 if collab_df['collab_score'].max() != 0 else 0
            rated_products = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
            collab_df = collab_df[~collab_df['product_id'].isin(rated_products)]
            recs = recs.merge(collab_df, on='product_id', how='left')
            recs['score'] += recs['collab_score'].fillna(0)
        cursor.execute("SELECT query FROM search_log WHERE user_id = %s ORDER BY created_at DESC LIMIT 1;", (user_id,))
        result = cursor.fetchone()
        suggestion = None
        if result:
            query = result[0]
            search_recs, suggestion = content_based_recommendations(query, products, n=10)
            search_scores = {r.product_id: r.predicted_rating * 0.2 for r in search_recs}
            recs['score'] += recs['product_id'].map(search_scores).fillna(0)
        pop_weight = 0.2 if is_new_user else 0.1
        pop_recs = popular_products[['product_id', 'score']].copy()
        pop_recs['pop_score'] = pop_recs['score'] * pop_weight
        recs = recs.merge(pop_recs[['product_id', 'pop_score']], on='product_id', how='left')
        recs['score'] += recs['pop_score'].fillna(0)
        user_profile = user_profiles[user_profiles['user_id'] == user_id]
        if not user_profile.empty:
            gender = user_profile['gender'].iloc[0].lower()
            age = user_profile['age'].iloc[0]
            if gender == 'female':
                recs['score'] += recs['category_name'].str.lower().isin(['rings', 'earrings', 'necklace', 'maang tikka', 'nose pins', 'waist chains']).astype(int) * 0.2
            elif gender == 'male':
                recs['score'] += recs['category_name'].str.lower().isin(['bracelets', 'brooches']).astype(int) * 0.2
            if 13 <= age <= 25:
                recs['score'] += recs['category_name'].str.lower().isin(['bracelets', 'anklets', 'nose pins']).astype(int) * 0.1
            elif 26 <= age <= 40:
                recs['score'] += recs['category_name'].str.lower().isin(['rings', 'earrings', 'necklace']).astype(int) * 0.1
            elif age > 40:
                recs['score'] += recs['category_name'].str.lower().isin(['bangles', 'waist chains', 'brooches']).astype(int) * 0.1
        cursor.execute("SELECT product_id FROM recommended_products WHERE user_id = %s ORDER BY updated_at DESC LIMIT 10;", (user_id,))
        recommended_products = pd.DataFrame(cursor.fetchall(), columns=['product_id'])
        if not recommended_products.empty:
            recs['score'] += recs['product_id'].isin(recommended_products['product_id']).astype(int) * 0.3
        recs = recs[recs['price'].between(price_range_min, price_range_max)]
        recs = recs.sort_values('score', ascending=False).head(n)
        if len(recs) < n:
            recs = pd.concat([recs, products.head(n - len(recs))]).drop_duplicates().head(n)
        recs['score'] = recs['score'].fillna(0.1)
        recs['predicted_rating'] = recs['collab_score'].fillna(3.0) * 5 if 'collab_score' in recs.columns else 3.0
        recommendations = [
            Recommendation(
                product_id=int(row['product_id']),
                category=str(row['category_name']),
                material=str(row['material']),
                price=float(row['price']),
                predicted_rating=float(row['predicted_rating'])
            )
            for _, row in recs.iterrows()
        ]
        execute_values(
            cursor,
            "INSERT INTO recommended_products (id, user_id, product_id) VALUES %s ON CONFLICT DO NOTHING;",
            [(str(uuid.uuid4()), user_id, r.product_id) for r in recommendations]
        )
        conn.commit()
        logger.info(f"Returning {len(recommendations)} recs for user {user_id}")
        return recommendations, suggestion
    except Exception as e:
        logger.error(f"Error for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.get("/")
async def root():
    return FileResponse("index.html")  # Serve index.html directly from project root

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: str, api_key: str = Depends(verify_api_key)):
    if user_factors is None or item_factors is None or products is None or user_profiles is None or global_popularity is None:
        if not load_data():
            logger.warning(f"No data loaded, returning empty recommendations for {user_id}")
            return {"recommendations": [], "suggestion": "No products available, please try again later"}
    recommendations, suggestion = recommend_products(user_id, user_factors, item_factors, products, user_price_preferences, popular_products, user_item_matrix, user_profiles, global_popularity)
    return {"recommendations": recommendations, "suggestion": suggestion}

@app.get("/recommend/activity/{user_id}", response_model=RecommendationResponse)
async def get_activity_recommendations(user_id: str, api_key: str = Depends(verify_api_key)):
    if products is None:
        if not load_data():
            return {"recommendations": [], "suggestion": "No products available"}
    recommendations, suggestion = activity_based_recommendations(user_id, products)
    return {"recommendations": recommendations, "suggestion": suggestion}

@app.post("/search", response_model=RecommendationResponse)
async def search_products(search: SearchRequest, api_key: str = Depends(verify_api_key)):
    try:
        logger.info(f"Search for user {search.user_id}: {search.query}")
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO search_log (id, user_id, query) VALUES (gen_random_uuid(), %s, %s);", (search.user_id, search.query))
        conn.commit()
        if products is None:
            if not load_data():
                raise HTTPException(status_code=500, detail="Failed to load data")
        recommendations, suggestion = content_based_recommendations(search.query, products)
        return {"recommendations": recommendations, "suggestion": suggestion}
    except Exception as e:
        logger.error(f"Search error for {search.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.post("/log_activity")
async def log_activity(activity: ActivityRequest, api_key: str = Depends(verify_api_key)):
    try:
        logger.info(f"Logging activity: request_body={activity.dict()}")
        if not activity.user_id or not activity.user_id.strip():
            logger.error("Validation failed: user_id is empty or invalid")
            raise HTTPException(status_code=422, detail="Invalid user_id: must be a non-empty string")
        
        normalized_activity_type = activity.activity_type.replace(' ', '_').lower()
        valid_activity_types = ['view', 'click', 'add_to_cart', 'wishlist']
        rating_pattern = r'^rating_([1-5])$'
        rating_match = re.match(rating_pattern, normalized_activity_type)
        
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        if not activity.user_id.startswith('anon_'):
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s;", (activity.user_id,))
            if not cursor.fetchone():
                logger.error(f"User {activity.user_id} not found")
                raise HTTPException(status_code=400, detail=f"User {activity.user_id} not found")
        
        product_name = None
        product_price = None
        if activity.product_id is not None:
            if products is None:
                if not load_data():
                    logger.error("Failed to load product data")
                    raise HTTPException(status_code=500, detail="Failed to load data")
            product = products[products['product_id'] == activity.product_id]
            if product.empty:
                logger.error(f"Invalid product_id: {activity.product_id}")
                raise HTTPException(status_code=400, detail=f"Invalid product_id: {activity.product_id}")
            product_name = product['category_name'].iloc[0]
            product_price = float(product['price'].iloc[0])
        
        if rating_match:
            rating = int(rating_match.group(1))
            if activity.product_id is None:
                logger.error("Rating requires a product_id")
                raise HTTPException(status_code=400, detail="Rating requires a product_id")
            cursor.execute(
                "INSERT INTO interactions (id, user_id, product_id, rating) VALUES (gen_random_uuid(), %s, %s, %s);",
                (activity.user_id, activity.product_id, rating)
            )
            logger.info(f"Logged rating {rating} for user {activity.user_id}, product_id {activity.product_id}, product_name {product_name}")
        elif normalized_activity_type == 'click' and activity.product_id is not None:
            cursor.execute(
                "INSERT INTO order_item (id, user_id, product_id, price) VALUES (gen_random_uuid(), %s, %s, %s);",
                (activity.user_id, activity.product_id, product_price)
            )
            cursor.execute(
                "INSERT INTO user_activity_log (id, user_id, product_id, activity_type) VALUES (gen_random_uuid(), %s, %s, %s);",
                (activity.user_id, activity.product_id, normalized_activity_type)
            )
            logger.info(f"Logged order and click for user {activity.user_id}, product_id {activity.product_id}, product_name {product_name}")
        elif normalized_activity_type in valid_activity_types:
            cursor.execute(
                "INSERT INTO user_activity_log (id, user_id, product_id, activity_type) VALUES (gen_random_uuid(), %s, %s, %s);",
                (activity.user_id, activity.product_id, normalized_activity_type)
            )
            logger.info(f"Logged {normalized_activity_type} for user {activity.user_id}, product_id {activity.product_id}, product_name {product_name if product_name else 'None'}")
        else:
            logger.error(f"Validation failed: invalid activity_type '{activity.activity_type}' (normalized: '{normalized_activity_type}')")
            raise HTTPException(status_code=400, detail=f"Invalid activity_type: {activity.activity_type}. Must normalize to one of {valid_activity_types} or rating_[1-5]")
        
        conn.commit()
        return {"status": "Activity logged"}
    except ValidationError as ve:
        logger.error(f"Pydantic validation error: {ve.errors()}")
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    except Exception as e:
        logger.error(f"Activity log error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.get("/user_activity/{user_id}", response_model=List[UserActivityResponse])
async def get_user_activity(user_id: str, api_key: str = Depends(verify_api_key)):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT ual.user_id, ual.product_id, COALESCE(c.category_name, 'Unknown') AS product_name, ual.activity_type, ual.created_at
            FROM user_activity_log ual
            LEFT JOIN products p ON ual.product_id = p.product_id
            LEFT JOIN category c ON p.category_id = c.category_id
            WHERE ual.user_id = %s
            ORDER BY ual.created_at DESC;
            """,
            (user_id,)
        )
        activities = cursor.fetchall()
        if not activities:
            logger.info(f"No activities found for user {user_id}")
            return []
        response = [
            UserActivityResponse(
                user_id=row[0],
                product_id=row[1],
                product_name=row[2] if row[2] != 'Unknown' else None,
                activity_type=row[3],
                created_at=str(row[4])
            )
            for row in activities
        ]
        logger.info(f"Retrieved {len(response)} activities for user {user_id}")
        return response
    except Exception as e:
        logger.error(f"Error fetching activities for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.get("/product/{product_id}", response_model=ProductResponse)
async def get_product_details(product_id: int, user_id: str = "100", api_key: str = Depends(verify_api_key)):
    try:
        if products is None:
            if not load_data():
                raise HTTPException(status_code=500, detail="Failed to load data")
        product = products[products['product_id'] == product_id]
        if product.empty:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        return ProductResponse(
            product_id=int(product['product_id'].iloc[0]),
            category=str(product['category_name'].iloc[0]),
            material=str(product['material'].iloc[0]),
            price=float(product['price'].iloc[0]),
            user_id=user_id
        )
    except Exception as e:
        logger.error(f"Error fetching product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/product/{product_id}/recommend", response_model=RecommendationResponse)
async def get_product_recommendations(product_id: int, user_id: str = "100", api_key: str = Depends(verify_api_key)):
    if products is None or user_profiles is None:
        if not load_data():
            return {"recommendations": [], "suggestion": "No products available"}
    recommendations, suggestion = product_based_recommendations(product_id, user_id, products, user_profiles)
    return {"recommendations": recommendations, "suggestion": suggestion}

@app.post("/signup", response_model=dict)
async def signup(signup: SignupRequest, api_key: str = Depends(verify_api_key)):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute("SELECT email, mobile FROM users WHERE email = %s OR mobile = %s;", (signup.email, signup.mobile))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email or mobile number already registered")
        user_id = str(np.random.randint(1000, 9999))
        cursor.execute(
            "INSERT INTO users (user_id, name, age, gender, email, mobile, password) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING user_id, name, age, gender, email, mobile;",
            (user_id, signup.name, signup.age, signup.gender, signup.email, signup.mobile, signup.password)
        )
        user_data = cursor.fetchone()
        conn.commit()
        logger.info(f"User {user_id} signed up successfully")
        return {
            "status": "User registered",
            "user_id": user_data[0],
            "name": user_data[1],
            "age": user_data[2],
            "gender": user_data[3],
            "email": user_data[4],
            "mobile": user_data[5]
        }
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.post("/login", response_model=LoginResponse)
async def login(login: LoginRequest, api_key: str = Depends(verify_api_key)):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, password FROM users WHERE email = %s;", (login.email,))
        result = cursor.fetchone()
        if not result or result[1] != login.password:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        user_id = result[0]
        logger.info(f"User {user_id} logged in successfully")
        return {"user_id": user_id}
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "API running"}

if __name__ == '__main__':
    logger.info("Starting server: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 8000)))