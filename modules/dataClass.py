import os
import copy
import numpy as np 
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler

#functions in the class:
#0a. get_features(self) -> list
#0b. get_copy_of_original_data(self) -> pd.DataFrame
#1. get_data_without_repeated_songs(self, input_data:pd.DataFrame = None) -> pd.DataFrame
#2. get_data_with_groupped_ranking(self, input_data:pd.DataFrame = None, group_size: int = 10) -> pd.DataFrame
#3. get_mean_and_std(self, input_data:pd.DataFrame = None, group_size: int = 10) -> tuple
#4. get_normalized_data(self, input_data:pd.DataFrame = None, remove_repeated_songs_for_mean_and_std:bool = True) -> pd.DataFrame
#5. get_normalized_data_0_to_1(self, input_data:pd.DataFrame = None) -> pd.DataFrame
#6a. get_outliers_IQR(self, input_data:pd.DataFrame = None) -> pd.DataFrame
#6b. remove_outliers_IQR(self, input_data:pd.DataFrame = None) -> pd.DataFrame:


class dataClass():

    #initalization of the class
    def __init__(self) -> None:

        #location of the origianl data
        self.DATA_PATH = os.path.join(os.getcwd(), 'datasets', 'Spotify_Dataset_V3.csv')
        #all features
        self.ALL_FATURES = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence']
        #data from the original dataset
        self.SPOTIFY_ORIGINAL_DATA = pd.read_csv(self.DATA_PATH, delimiter = ';')
        

    def get_features(self) -> list:
        return self.ALL_FATURES
        
    def get_copy_of_original_data(self) -> pd.DataFrame:
        return self.SPOTIFY_ORIGINAL_DATA.copy(deep=True)
    
    #if a song was done by multiple artists take only one istance of this song
    def get_data_without_repeated_songs(self, input_data:pd.DataFrame = None) -> pd.DataFrame:
        
        if input_data is None:
            input_data = self.SPOTIFY_ORIGINAL_DATA
            
        spotify_data_with_songs_only = input_data.copy(deep = True)
        spotify_data_with_songs_only = spotify_data_with_songs_only[spotify_data_with_songs_only["# of Artist"] == "Artist 1"]
        
        return spotify_data_with_songs_only
    
    #function to change rank into groups (e.g. if group_size is 10, new rank 1 will be top 10,
    # rank 2 will be 11-20 and so on)
    def get_data_with_groupped_ranking(self, input_data:pd.DataFrame = None, group_size: int = 10) -> pd.DataFrame:
        
        if input_data is None:
           input_data = self.SPOTIFY_ORIGINAL_DATA
           
        spotify_data_groupped_ranks = input_data.copy(deep = True)
        spotify_data_groupped_ranks["Rank"] = ((spotify_data_groupped_ranks["Rank"]/group_size).apply(np.ceil)).astype(int)
        
        return spotify_data_groupped_ranks
    
    #get means and stds of the "input_data" (if it is not supplied, calculations will be done on the whole dataset)
    def get_mean_and_std(self, input_data:pd.DataFrame = None, group_size: int = 10) -> tuple:
        
        if input_data is None:
           input_data = self.SPOTIFY_ORIGINAL_DATA
        
        list_of_means = list()
        list_of_stds = list()
        
        for feature in self.ALL_FATURES:
            list_of_means.append(input_data[feature].mean())
            list_of_stds.append(input_data[feature].std())
        
        return (list_of_means, list_of_stds)
    
    
    #normalize data (remove_repeated_songs_for_mean_and_std set to True to skip repeated songs)
    def get_normalized_data(self, input_data:pd.DataFrame = None, remove_repeated_songs_for_mean_and_std:bool = True) -> pd.DataFrame:
        
        if input_data is None:
           input_data = self.SPOTIFY_ORIGINAL_DATA
           
        normalized_data = input_data.copy(deep = True)
        
        if remove_repeated_songs_for_mean_and_std:
            data_to_get_means_and_std = self.get_data_without_repeated_songs(input_data= input_data)
        else:
            data_to_get_means_and_std = input_data
        
        means, STDs = self.get_mean_and_std(data_to_get_means_and_std)
        for i, feature in enumerate(self.ALL_FATURES):
            normalized_data[feature] = (normalized_data[feature] - means[i]) / STDs[i]
        
        return normalized_data
    

    def get_normalized_data_0_to_1(self, input_data:pd.DataFrame = None) -> pd.DataFrame:
        
        if input_data is None:
           input_data = self.SPOTIFY_ORIGINAL_DATA
           
        normalized_data = input_data.copy(deep = True)
        
        scaler = MinMaxScaler()
        for category in self.ALL_FATURES:
            normalized_data[category] = scaler.fit_transform(normalized_data[[category]])
        
        return normalized_data
    

    #find the outlier in the data and replace it with the mean value
    def get_outliers_IQR(self, input_data:pd.DataFrame = None) -> pd.DataFrame:
        if input_data is None:
            input_data = self.SPOTIFY_ORIGINAL_DATA
        data = input_data.copy(deep=True)
        features_list = ['Danceability','Energy','Loudness','Speechiness','Acousticness','Instrumentalness']
        for feature in features_list:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))
            mean = data.loc[~outliers, feature].mean()
            data.loc[outliers, feature] = mean
        return data
    

    #find the outlier in the data and remove it
    def remove_outliers_IQR(self, input_data:pd.DataFrame = None) -> pd.DataFrame:
        if input_data is None:
            input_data = self.SPOTIFY_ORIGINAL_DATA
        data = input_data.copy(deep=True)
        features_list = ['Danceability','Energy','Loudness','Speechiness','Acousticness','Instrumentalness']
        for feature in features_list:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))
            data = data[~outliers]
        return data

    def get_data_for_ml(self, input_data:pd.DataFrame = None) -> pd.DataFrame:

        # get normalised data and remove duplicates
        df = self.get_normalized_data(input_data=input_data,remove_repeated_songs_for_mean_and_std=True)

        # Remove outliers or not:
        df = self.remove_outliers_IQR(df)
        
        columns_to_exclude = ['Rank','Title','Artists','# of Artist','Artist (Ind.)','# of Nationality','Nationality','Continent','Points (Total)','Points (Ind for each Artist/Nat)' ]  # Add the column names you want to exclude
        df = df.drop(columns=columns_to_exclude).copy()


        # find earliest date for each song
        earliest_dates = df.groupby('id')['Date'].min().reset_index()
        df = pd.merge(df, earliest_dates, on=['id'], suffixes=('', '_earliest'))


        df['Date'] = pd.to_datetime(df['Date'])
        df['Date_earliest'] = pd.to_datetime(df['Date_earliest'])

        # gets total amount of identical songs
        df['song_count'] = df.groupby('id')['id'].transform('count')
        i_song_df= df.drop_duplicates(subset='id')

        only_dates = df[['Date','id']]

        df_a = i_song_df.copy()
        df_b = only_dates.copy()

        # Function to count rows in B with dates newer than A's 'e_date' for different 'id'
        def count_newer_dates_diff_id(row):
            id_diff = (df_b['id'] != row['id']) & (df_b['Date'] > row['Date_earliest'])
            print(id_diff)
            return len(df_b[id_diff])

        df_a['newer_date_count_diff_id'] = df_a.apply(count_newer_dates_diff_id, axis=1)
        df_a['ratio'] = df_a['song_count'] / (df_a['song_count']+df_a['newer_date_count_diff_id'])

        df_a.to_csv('data_with_outliers.csv', index=False)





