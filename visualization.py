#%% Imports 

import os
import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pandas.api.types import CategoricalDtype
#%matplotlib inline


#%% Loading the data

data_path = os.path.join(os.getcwd(), 'datasets', 'Spotify_Dataset_V3.csv')
spotifybase = pd.read_csv(data_path, delimiter = ';')

spotifybase.head(5)
# %% DATA CLEANING
spotifybase.drop(["Song URL"], axis=1, inplace=True)
#%% DATA NORMALIZATION using the Z-SCORE METHOD
norm_base = spotifybase.copy() 
# apply normalization techniques 
for column in ["Danceability","Energy","Loudness","Speechiness","Acousticness","Instrumentalness","Valence"]: 
    norm_base[column] = (norm_base[column] -norm_base[column].mean()) / norm_base[column].std() 


norm_base['Date'] = pd.to_datetime(norm_base['Date'])

# %% plotting traits by year
yearly_data = norm_base.groupby(norm_base['Date'].dt.year)[['Danceability', 'Energy', 'Loudness','Speechiness','Acousticness','Instrumentalness','Valence']].mean()


# Create the line chart
plt.figure(figsize=(10, 6))

# Plot Traits 
for trait in ['Danceability', 'Energy', 'Loudness','Speechiness','Acousticness','Instrumentalness','Valence']:
    plt.plot(yearly_data.index, yearly_data[trait], label=trait, marker='o', linestyle='-')

plt.title('Yearly Trends for Traits')
plt.xlabel('Year')
plt.ylabel('Values')
plt.grid(True)

# Optionally, customize the x-axis labels to make them more readable
plt.xticks(yearly_data.index)

# Add a legend to distinguish the lines
plt.legend()
# Show the plot
plt.show()


# %% CONTINENT PLOTTING


# Group the data by year and continent and calculate the count of songs for each combination
yearly_continent_counts = norm_base.groupby([norm_base['Date'].dt.year, 'Continent'])['id'].count().unstack(fill_value=0)
yearly_percentages = yearly_continent_counts.div(yearly_continent_counts.sum(axis=1), axis=0) * 100

# Create the line graph
plt.figure(figsize=(10, 6))

# Plot each continent as a line
for continent in yearly_percentages.columns:
    plt.plot(yearly_percentages.index, yearly_percentages[continent], label=continent, marker='o', linestyle='-')

plt.title('Total Number of Songs by Continent Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Number of Songs')
plt.grid(True)

# Optionally, customize the x-axis labels to make them more readable
plt.xticks(yearly_percentages.index)

# Add a legend to distinguish the lines
plt.legend()

# Show the plot
plt.show()
# %% NATIONALITY PLOTTING
# Group the data by year and continent and calculate the count of songs for each combination
yearly_national_counts = norm_base.groupby([norm_base['Date'].dt.year, 'Nationality'])['id'].count().unstack(fill_value=0)
yearly_national_counts = yearly_national_counts .div(yearly_national_counts.sum(axis=1), axis=0) * 100



# Calculate the total number of songs for each continent
nationality_totals = yearly_national_counts.sum()

# Select the top 3 continents with the highest total number of songs
top_10_nationalities = nationality_totals.nlargest(10).index

# Filter the data to keep only the top 3 continents
filtered_data = yearly_national_counts[top_10_nationalities]


# Create the line graph
plt.figure(figsize=(10, 6))

# Plot each continent as a line
for nationality in filtered_data.columns:
    plt.plot(yearly_national_counts.index, yearly_national_counts[nationality], label=nationality, marker='o', linestyle='-')

plt.title('Percentage of Total Songs belonging to each Nationality Over the Years')
plt.xlabel('Year')
plt.ylabel('Perecentage of Total songs')
plt.grid(True)

# Optionally, customize the x-axis labels to make them more readable
plt.xticks(yearly_continent_counts.index)

# Add a legend to distinguish the lines
plt.legend()

# Show the plot
plt.show()

# %% cheching inconsistencies
yearly_song_counts = norm_base.groupby(norm_base['Date'].dt.year)['id'].count()
print(yearly_song_counts)
# %% Popularity of Popular artists over the years
yearly_artist_counts = norm_base.groupby([norm_base['Date'].dt.year, 'Artists'])['id'].count().unstack(fill_value=0)
yearly_artist_counts = yearly_artist_counts.div(yearly_artist_counts.sum(axis=1), axis=0) * 100


artist_totals = yearly_artist_counts.sum() # Calculate the total number of songs for each artist

# Select the top artists with the highest total number of songs
top_10_artists = artist_totals.nlargest(10).index

# Filter the data to keep only the top 3 continents
filtered_data = yearly_artist_counts[top_10_artists]


# Create the line graph
plt.figure(figsize=(10, 6))

# Plot each continent as a line
for Artists in filtered_data.columns:
    plt.plot(yearly_artist_counts.index, yearly_artist_counts[Artists], label=Artists, marker='o', linestyle='-')

plt.title('Total Number of Songs by Nationality Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Number of Songs')
plt.grid(True)

# Optionally, customize the x-axis labels to make them more readable
plt.xticks(yearly_artist_counts.index)

# Add a legend to distinguish the lines
plt.legend()

# Show the plot
plt.show()

# %%  THE NEW DATASET 

# taking the database that has normalized the traits values 
df =norm_base.copy()

df['Date'] = pd.to_datetime(df['Date']) 

# excluding duplicates of songs
df = df[df["# of Artist"] == "Artist 1"]

# drop columns we dont need
columns_to_exclude = ['Rank','Title','Artists','# of Artist','Artist (Ind.)','# of Nationality','Nationality','Continent','Points (Ind for each Artist/Nat)' ]  # Add the column names you want to exclude
df = df.drop(columns=columns_to_exclude).copy()


# find earliest date for each song and then create a new column with that info 
earliest_dates = df.groupby('id')['Date'].min().reset_index()
df = pd.merge(df, earliest_dates, on=['id'], suffixes=('', '_earliest'))

# making sure dates are in date format 
df['Date'] = pd.to_datetime(df['Date'])
df['Date_earliest'] = pd.to_datetime(df['Date_earliest'])


# gets total amount of songs per song and makes a new column 
df['song_count'] = df.groupby('id')['id'].transform('count')

# calculates total amount of points for each song 
total_points_per_song = df.groupby('id')['Points (Total)'].sum().reset_index()

# Drop dupliactes of songs with the same id 
df= df.drop_duplicates(subset='id')

# merge 'total amount of points for each song' with original dataset
df = pd.merge(df, total_points_per_song, on='id', how='left', suffixes=('', '_grouped'))

# get the most recent date from the dataset
actual_date = pd.to_datetime('2023-05-29') 

# create a new column that represnts how many days a song has been on this dataset, from its first appearance 
df['date_difference'] = actual_date - df['Date_earliest']

