import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from IPython.display import display

# Ensuring reproducibility
np.random.seed(42)

# Data Generation
students = range(1, 301)
degrees = np.random.choice(['Business', 'Law', 'Engineering', 'Design'], 300)
genders = np.random.choice(['Male', 'Female'], 300)
lunch_spots = np.random.choice(
    ['Honest Greens', 'Makkila', 'Pancake House', 'Five Guys', 'Starbucks', 'IE Cafeteria', 'New Spot'], 300)
time_spent = np.random.normal(loc=35, scale=15, size=300).astype(int)  # Average time spent in minutes
ratings = np.random.randint(1, 6, size=300)  # Ratings from 1 to 5
price_ranges = np.random.choice(['Low', 'Medium', 'High'], 300, p=[0.2, 0.6, 0.2])
assortment = np.random.choice(['Limited', 'Diverse', 'Very Diverse'], 300, p=[0.3, 0.5, 0.2])
freshness = np.random.choice(['Average', 'Fresh', 'Very Fresh'], 300, p=[0.2, 0.5, 0.3])
instagramable = np.random.choice([True, False], 300, p=[0.6, 0.4])
atmosphere = np.random.choice(['Cozy', 'Modern', 'Elegant', 'Casual'], 300)
waiting_time = np.random.normal(loc=10, scale=5, size=300).astype(int)  # Average waiting time in minutes

# DataFrame creation
df = pd.DataFrame({
    'Student_ID': students,
    'Degree': degrees,
    'Gender': genders,
    'Lunch_Spot': lunch_spots,
    'Time_Spent': time_spent,
    'Rating': ratings,
    'Price_Range': price_ranges,
    'Assortment': assortment,
    'Freshness': freshness,
    'Instagramable': instagramable,
    'Atmosphere': atmosphere,
    'Waiting_Time': waiting_time
})

# Print the HTML table
print(df)
