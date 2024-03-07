# I am

import pip
import pyarrow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ensuring reproducibility
np.random.seed(42)

# Data Generation
students = range(1, 301)
degrees = np.random.choice(['Business', 'Law', 'Engineering', 'Design'], 300)
genders = np.random.choice(['Male', 'Female'], 300)
lunch_spots = np.random.choice(['Honest Greens', 'Makkila', 'Pancake House', 'Five Guys', 'Starbucks', 'IE Cafeteria', 'New Spot'], 300)
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

# Advanced Visualization: Heatmap for Correlation Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Data Preprocessing for Clustering
features = df[['Time_Spent', 'Rating', 'Waiting_Time']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Performing K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)
# Assuming the other parts of your DataFrame df are correctly defined
# Adding the 'Dietary_Preferences' column with a mock distribution
df['Dietary_Preferences'] = np.random.choice(['No Preference', 'Vegetarian', 'Vegan', 'Gluten-Free'], size=300, p=[0.5, 0.2, 0.2, 0.1])

# Calculate the number of students by dietary preference for each lunch spot
dietary_lunch_spot_counts = df.groupby(['Lunch_Spot', 'Dietary_Preferences']).size().unstack(fill_value=0)

# Plot a stacked bar chart

# Group by 'Lunch_Spot' and 'Dietary_Preferences', then count the number of students in each group
dietary_lunch_spot_counts = df.groupby(['Lunch_Spot', 'Dietary_Preferences']).size().unstack(fill_value=0)

# Plotting the stacked bar chart
dietary_lunch_spot_counts.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
plt.title('Popularity of Lunch Spots by Dietary Preferences')
plt.xlabel('Lunch Spot')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.legend(title='Dietary Preferences', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Cross-Analysis: Price Range Preference by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Price_Range', hue='Gender', data=df, palette='Pastel2')
plt.title('Price Range Preference by Gender')
plt.show()

# Insights & Recommendations based on the analysis (not executable, conceptual)