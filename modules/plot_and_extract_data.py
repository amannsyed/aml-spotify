import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'modules'))

import copy
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dataClass


#functions in the class:
#1. plot_mean_vals_of_features_over_ranks(self, year:str = None, size_of_groups:int=1) -> None
#2. plot_features_over_time(self, mode = "all", scaling:str = "standard") -> None
#3. points_per_author_type_per_year(self, author_type: str) -> pd.DataFrame
#4. points_per_author_type_per_month(self, author_type: str) -> pd.DataFrame:
#5. convert_points_in_a_file_to_percentages(self, file: str) -> pd.DataFrame
#6. drop_authors_with_low_percentage(self, source_file: str, total_mode: bool, threshold: float, merge_not_passed:bool = True) -> pd.DataFrame:


class plot_and_extract_data():
    
    def __init__(self):
        self.PLOTS_PATH = os.path.join(os.getcwd(), 'plots')
        self.DATA_OBJECT = dataClass.dataClass()
        self.lw = 2
        if not os.path.exists(self.PLOTS_PATH):
            os.mkdir(self.PLOTS_PATH)
        self.EXTRACTED_DATA_PATH =os.path.join(os.getcwd(), "extracted_data")
        if not os.path.exists(self.EXTRACTED_DATA_PATH):
            os.mkdir(self.EXTRACTED_DATA_PATH)
            

    #This function plots mean values of features over ranks
    #if year is not set the whole dataset will be taken 
    #the data that is plotted is also saved into a csv file      
    def plot_mean_vals_of_features_over_ranks(self, year:str = None, size_of_groups:int=1) -> None:
        plots_loc = os.path.join(self.PLOTS_PATH, "feature_over_ranks")
        if not os.path.exists(plots_loc):
            os.mkdir(plots_loc)
        
        fearures = self.DATA_OBJECT.get_features()
        data_to_plot = self.DATA_OBJECT.get_data_without_repeated_songs()
        data_to_plot = self.DATA_OBJECT.get_normalized_data(input_data=data_to_plot)
        data_to_plot = self.DATA_OBJECT.get_data_with_groupped_ranking(input_data = data_to_plot, group_size= size_of_groups)
        
        number_of_groups = data_to_plot["Rank"].unique().size
        X_rank = np.arange(number_of_groups) + 1
        Y_combined = list()
        
        csv_to_write = os.path.join(self.EXTRACTED_DATA_PATH, "combined_means_over_ranks")
        if not os.path.exists(csv_to_write):
            os.mkdir(csv_to_write)
        csv_to_write = os.path.join(csv_to_write, f"group_size_{size_of_groups}")
        if not os.path.exists(csv_to_write):
            os.mkdir(csv_to_write)
        
        if year:
            data_to_plot = data_to_plot[data_to_plot["Date"].str.contains(str(year), case = False, na= False)]
            plots_loc = plots_loc +os.sep+ f"feature_over_ranks_{year}"
            csv_to_write = os.path.join(csv_to_write, f"combined_features_mean_{year}.csv")
        else:
            plots_loc = plots_loc +os.sep+ f"feature_over_ranks_all_time"
            csv_to_write = os.path.join(csv_to_write, f"combined_features_mean__all_time.csv")
    
        if not os.path.exists(plots_loc):
            os.mkdir(plots_loc)
        
        
        for i, feature in enumerate(fearures):
            plt.figure()
            print("plotting in progress...")
            Y_mean = np.zeros(number_of_groups)
            Y_std = np.zeros(number_of_groups)
            Y_median = np.zeros(number_of_groups)


            for i in range(number_of_groups):
                filtered_data = data_to_plot[data_to_plot["Rank"] == i+1]
                Y_mean[i] = filtered_data[feature].mean()
                Y_std[i] = filtered_data[feature].std()
                Y_median[i] = filtered_data[feature].median()

            Y_combined.append(copy.deepcopy(Y_mean))
                
            plt.plot(X_rank, Y_mean, color = "red", label= "mean")
            plt.plot(X_rank, Y_std, color = "green", label = "std")
            plt.plot(X_rank, Y_median, color = "yellow", label = "median")
            
            plt.legend(loc='best')
            plt.grid()
            plt.xlabel("Rank")
            plt.ylabel(feature)
            plt.savefig(os.path.join(plots_loc, feature))
            plt.close()
            
            

        plt.figure()
        for i in range(len(fearures)):
            plt.plot(X_rank, Y_combined[i], label= fearures[i], marker='o', linestyle='-')
        
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel("Rank")
        plt.ylabel("mean values over ranks")
        
        path_to_save = plots_loc +os.sep+ f"combined_means"
        plt.savefig(path_to_save)
        plt.close()
        
        dataFrame_to_save = pd.DataFrame(columns=["Rank_group"] + self.DATA_OBJECT.get_features())
        for row in range(len(Y_combined[i])):
            dataFrame_to_save.loc[row+1] = [int(row+1)] + [Y_combined[i][row] for i in range(len(self.DATA_OBJECT.get_features()))]
        
        dataFrame_to_save.to_csv(csv_to_write, index=False, header=True)
        
    
    #This function plots mean values of features over time
    #both daily and yearly data are plotted
    #the data that is plotted is also saved into a csv file
    def plot_features_over_time(self, mode = "all", scaling:str = "standard") -> None:
        
        plots_loc = os.path.join(self.PLOTS_PATH, f"features_over_time_{scaling}_normalization")
        csv_loc = os.path.join("extracted_data", f"features_over_time_{scaling}_normalization")
        if not os.path.exists(csv_loc):
            os.mkdir(csv_loc)
        if not os.path.exists(plots_loc):
            os.mkdir(plots_loc)
            
        if not os.path.exists(plots_loc +os.sep+ "daily"):
            os.mkdir(plots_loc +os.sep+ "daily")
        if not os.path.exists(plots_loc +os.sep+ "yearly"):
            os.mkdir(plots_loc +os.sep+ "yearly")
        
        if not os.path.exists(csv_loc +os.sep+ "daily"):
            os.mkdir(csv_loc +os.sep+ "daily")
        if not os.path.exists(csv_loc +os.sep+ "yearly"):
            os.mkdir(csv_loc +os.sep+ "yearly")
        
        plots_loc_yearly = plots_loc +os.sep+ "yearly"
        plots_loc_daily  = plots_loc +os.sep+ "daily"
        csv_loc_yearly   = csv_loc +os.sep+ "yearly"
        csv_loc_daily    = csv_loc +os.sep+ "daily"
        
        spotify_data_copy = self.DATA_OBJECT.get_data_without_repeated_songs()
        #
        if scaling == "standard":
            spotify_data_copy = self.DATA_OBJECT.get_normalized_data(input_data= spotify_data_copy, remove_repeated_songs_for_mean_and_std=True)
        else:
            spotify_data_copy = self.DATA_OBJECT.get_normalized_data_0_to_1(input_data= spotify_data_copy)
            
            
        combined_daily_data= list()
        combined_yearly_data = list()
        
        
        features = self.DATA_OBJECT.get_features()
        for fearure in features:

            data_len = spotify_data_copy["Date"].unique()
            if "2023" in spotify_data_copy['Date'].unique()[0]:
                data_len = data_len[::-1]
        
            
            #data_len= self.DATA_OBJECT.SPOTIFY_ORIGINAL_DATA["Date"].unique()[::-1]           
            X_time = np.arange(data_len.size)
            Y_mean = np.zeros(data_len.size)
            Y_std = np.zeros(data_len.size)
            Y_min = np.zeros(data_len.size)
            Y_max = np.zeros(data_len.size)
            Y_median = np.zeros(data_len.size)
            day_counter = 0
            for day in data_len:
                print(day)
                filtered_data = spotify_data_copy[spotify_data_copy["Date"] == day].copy(deep=True)
                if mode == "top":
                    filtered_data = filtered_data[filtered_data["Rank"] <= 10 ]
                elif mode == "bottom":
                    filtered_data = filtered_data[filtered_data["Rank"] >= 190 ]
                    
                Y_mean[day_counter] = filtered_data[fearure].mean()
                Y_std[day_counter] = filtered_data[fearure].std()
                Y_min[day_counter] = filtered_data[fearure].min()
                Y_max[day_counter] = filtered_data[fearure].max()
                Y_median[day_counter] = filtered_data[fearure].median()
                X_time[day_counter] = day_counter
                day_counter += 1
            combined_daily_data.append(copy.deepcopy(Y_mean))
                
            X_time_day = X_time
            #plt.figure(figsize= (15,10))
            plt.plot(X_time, Y_mean, color = "red", label= "mean", marker='o', linestyle='-', markersize=1)
            plt.plot(X_time, Y_std, color = "green", label = "std", marker='o', linestyle='-', markersize=1)
            #plt.plot(X_rank, Y_min, color = "brown", label= "min")
            #plt.plot(X_rank, Y_max, color = "blue", label = "max")
            plt.plot(X_time, Y_median, color = "yellow", label = "median", marker='o', linestyle='-', markersize=1)
            plt.legend(loc='upper left')
            plt.grid()
            plt.xlabel("time from 2017 to 2023")
            plt.ylabel(fearure)
            #plt.xticks([])
            #plt.show()
            path_to_save = plots_loc_daily +os.sep+ f"daily_{mode}_{fearure}"
            plt.savefig(path_to_save)
            plt.close()
            
            #plt.figure(figsize= (15,10))
            #for year in range(2017, 2023+1):    
                            
            ######
            data_len= len(range(2017, 2023+1))            
            X_time = np.arange(data_len)
            Y_mean = np.zeros(data_len)
            Y_std = np.zeros(data_len)
            Y_min = np.zeros(data_len)
            Y_max = np.zeros(data_len)
            Y_median = np.zeros(data_len)
            counter = 0
                
            for year in range(2017, 2023+1):
                #filtered_data = self.DATA_OBJECT.SPOTIFY_ORIGINAL_DATA_buckeded_rank[self.DATA_OBJECT.SPOTIFY_ORIGINAL_DATA_buckeded_rank["Date"] == i].copy(deep=True)
                filtered_data = spotify_data_copy[spotify_data_copy["Date"].str.contains(str(year), case = False, na= False)].copy(deep=True)
                if mode == "top":
                    filtered_data = filtered_data[filtered_data["Rank"] <= 10 ]
                elif mode == "bottom":
                    filtered_data = filtered_data[filtered_data["Rank"] >= 190 ]
                
                Y_mean[counter] = filtered_data[fearure].mean()
                Y_std[counter] = filtered_data[fearure].std()
                Y_min[counter] = filtered_data[fearure].min()
                Y_max[counter] = filtered_data[fearure].max()
                Y_median[counter] = filtered_data[fearure].median()
                X_time[counter] = counter
                counter += 1
            combined_yearly_data.append(copy.deepcopy(Y_mean))
                
            X_time_year = X_time
            plt.plot(X_time, Y_mean, color = "red", label= "mean", marker='o', linestyle='-')
            plt.plot(X_time, Y_std, color = "green", label = "std", marker='o', linestyle='-')
            #plt.plot(X_rank, Y_min, color = "brown", label= "min")
            #plt.plot(X_rank, Y_max, color = "blue", label = "max")
            plt.plot(X_time, Y_median, color = "yellow", label = "median", marker='o', linestyle='-')
            plt.legend(loc='upper left')
            plt.grid()
            plt.xlabel("time from 2017 to 2023")
            plt.ylabel(fearure)
            #plt.ylim(0, 1.1)
            #plt.xticks([])
            #plt.show()
            path_to_save = plots_loc_yearly +os.sep+ f"yearly_{mode}_{fearure}"
            plt.savefig(path_to_save)
            plt.close()
            ######

        #plt.figure(figsize= (15,10))
        for i in range(len(features)):
            plt.plot(X_time_day, combined_daily_data[i], label= features[i], marker='o', linestyle='-', markersize=1)
            
        plt.legend(loc='upper left')
        plt.grid()
        plt.xlabel("time from 2017 to 2023")
        plt.ylabel("values per day")
        #plt.ylim(0, 1.1)
        #plt.xticks([])
        #plt.show()
        path_to_save = plots_loc_daily +os.sep+ f"combined_daily_{mode}"
        plt.savefig(path_to_save)
        plt.close()
    
        #plt.figure(figsize= (15,10))
        for i in range(len(features)):
            plt.plot(X_time_year, combined_yearly_data[i], label= features[i], marker='o', linestyle='-')
            
        plt.legend(loc='upper left')
        plt.grid()
        plt.xlabel("time from 2017 to 2023")
        plt.ylabel("values per year")
        #plt.ylim(0, 1.1)
        #plt.xticks([])
        #plt.show()
        path_to_save = plots_loc_yearly +os.sep+ f"combined_yearly_{mode}"
        plt.savefig(path_to_save)
        plt.close()
        
        csv_combined_daily = pd.DataFrame(columns= ["day"] + features)
        csv_combined_yearly = pd.DataFrame(columns= ["year"] + features)
        
        for year in range(2017, 2023+1):
            csv_combined_yearly.loc[year] = [int(year)] + [combined_yearly_data[i][year-2017] for i in range(len(features))]
        for day in range(day_counter):
            csv_combined_daily.loc[day] = [int(day)] + [combined_daily_data[i][day] for i in range(len(features))]
        
        csv_combined_yearly.to_csv(os.path.join(csv_loc_yearly, f"combined_yearly_{mode}.csv"), sep=',', encoding='utf-8', index=False, header=True)
        csv_combined_daily.to_csv(os.path.join(csv_loc_daily, f"combined_daily_{mode}.csv"), sep=',', encoding='utf-8', index=False, header=True)
        
        
    #calculating the points gathered by authors in total and per year
    #(author type can be: an artist, country or continent; author type must be the same as the name of the column)
    #the data is also saved into a csv file
    def points_per_author_type_per_year(self, author_type: str) -> pd.DataFrame:
        new_data_loc = os.path.join(os.getcwd(), "extracted_data")
        if not os.path.exists(new_data_loc):
            os.mkdir(new_data_loc)
        new_data_loc  = os.path.join(new_data_loc, "points")
        if not os.path.exists(new_data_loc):
            os.mkdir(new_data_loc)   
            
        unique_authors = self.DATA_OBJECT.SPOTIFY_ORIGINAL_DATA[author_type].unique()
        unique_authors_points = pd.DataFrame(columns=["author", "points_sum"] + [f'points in {year}' for year in range(2017, 2023+1)])
        
        counter = 0
        for unique_author in unique_authors:
            counter += 1
            print(f"{counter}/{unique_authors.size}: calculating sum of points for {unique_author}")
            unique_authors_points.loc[counter] = [unique_author] + [float(0) for _ in (range(2017, 2023+1 +1))]
            data_for_author = self.DATA_OBJECT.SPOTIFY_ORIGINAL_DATA[self.DATA_OBJECT.SPOTIFY_ORIGINAL_DATA[author_type] == unique_author].copy(deep=True)
            #data_for_author["Date"] = data_for_author["Date"].astype(str)
            for year in range(2017, 2023+1):    
                data_for_author_per_year = data_for_author[data_for_author["Date"].str.contains(str(year), case = False, na= False)]
                unique_authors_points.at[counter, f'points in {year}'] = data_for_author_per_year["Points (Ind for each Artist/Nat)"].sum()
        
            unique_authors_points.at[counter, "points_sum"] = unique_authors_points.loc[counter, "points_sum":].sum()
        unique_authors_points.sort_values(by='points_sum', ascending=False, inplace=True)
        unique_authors_points.to_csv(os.path.join(new_data_loc, f"{author_type}.csv"), sep=',', encoding='utf-8', index=False, header=True)
        
        return unique_authors_points
    
    def points_per_author_type_per_month(self, author_type: str) -> pd.DataFrame:
        new_data_loc = os.path.join(os.getcwd(), "extracted_data")
        if not os.path.exists(new_data_loc):
            os.mkdir(new_data_loc)
        new_data_loc  = os.path.join(new_data_loc, "points")
        if not os.path.exists(new_data_loc):
            os.mkdir(new_data_loc)    
            
        unique_authors = self.DATA_OBJECT.SPOTIFY_ORIGINAL_DATA[author_type].unique()
        original_data = self.DATA_OBJECT.get_copy_of_original_data()
        #drop the day from a date
        original_data['Date'] = pd.to_datetime(original_data['Date'], format='%d/%m/%Y').dt.strftime('%m/%Y')
        months = original_data["Date"].unique()
        
        if "2023" in months[0]:
            months = months[::-1]
        unique_authors_points = pd.DataFrame(columns=["author", "points_sum"] + [f'points in {m}' for m in months])

        counter = 0
        for unique_author in unique_authors:
            counter += 1
            print(f"{counter}/{unique_authors.size}: calculating sum of points for {unique_author}")
            unique_authors_points.loc[counter] = [unique_author] + [float(0) for _ in (range(len(months)+1))]
            data_for_author = original_data[original_data[author_type] == unique_author].copy(deep=True)
            #data_for_author["Date"] = data_for_author["Date"].astype(str)
            for month in months:    
                data_for_author_per_month = data_for_author[data_for_author["Date"].str.contains(str(month), case = False, na= False)]
                unique_authors_points.at[counter, f'points in {month}'] = data_for_author_per_month["Points (Ind for each Artist/Nat)"].sum()
            unique_authors_points.at[counter, "points_sum"] = unique_authors_points.loc[counter, "points_sum":].sum()
        
        unique_authors_points.sort_values(by='points_sum', ascending=False, inplace=True)
        unique_authors_points.to_csv(os.path.join(new_data_loc, f"{author_type}_monthly.csv"), sep=',', encoding='utf-8', index=False, header=True)
        
        return unique_authors_points
    
    #this functions reads a csv document that has points per year for each author type
    #author's points in one year are divided by the sum of the points in that year
    #the data is saved into a csv file
    def convert_points_in_a_file_to_percentages(self, file: str) -> pd.DataFrame:
        data_from_file = pd.read_csv(file, delimiter = ',')
        data_with_percentages = data_from_file.copy(deep= True)
        dest_location = file.replace(".csv", "")
        dest_location += "_prercentage.csv"
        
        for column in data_from_file.columns:
            if "points" not in column:
                continue
            data_with_percentages[column] = data_with_percentages[column] / data_from_file[column].sum()
            
        data_with_percentages.to_csv(dest_location, index=False)
            
        return data_with_percentages
        
        
    #this function gets rid of authots that have low percentage of scored points
    #total_mode:
    #   set total_mode to True if you want to make reduction based on percentage of points scored since 2017
    #   set total_mode to False if you want to want to get rig of authors that did not reach the treshold in any year
    #merge_not_passed
    #   set merge_not_passed to True if you want to merge the authors that not passed the threshold into "others"
    def drop_authors_with_low_percentage(self, source_file: str, total_mode: bool, threshold: float, merge_not_passed:bool = True) -> pd.DataFrame:
        if "prercentage" not in source_file:
            print("enter a file with percentage values; exiting...")
            return
        
        data_from_file = pd.read_csv(source_file, delimiter = ',')
        data_with_reduced_artists = None
        
        if total_mode: 
            data_from_file_treshold_passed = data_from_file[data_from_file["points_sum"] >= threshold].copy(deep = True)
            data_from_file_below_treshold = data_from_file[data_from_file["points_sum"] < threshold].copy(deep = True)
        else:
            above_threshold_filter = (data_from_file.iloc[:, 1:]  >= threshold).any(axis=1)
            below_threshold_filter = (data_from_file.iloc[:, 1:]  < threshold).all(axis=1)
            data_from_file_treshold_passed = data_from_file[above_threshold_filter].copy(deep = True)
            data_from_file_below_treshold = data_from_file[below_threshold_filter].copy(deep = True)
        
        if merge_not_passed:
            not_passed_summed = pd.Series(["other"] +[data_from_file_below_treshold[column].sum() for column in data_from_file_below_treshold.columns[1:]], index=data_from_file_below_treshold.columns)
            data_from_file_treshold_passed = pd.concat([data_from_file_treshold_passed, not_passed_summed.to_frame().T], ignore_index=True)
            #data_from_file_treshold_passed = data_from_file_treshold_passed.append(data_from_file_treshold_passed, ignore_index=True)
        
        dest_location = source_file.replace(".csv", f"thr{threshold}_total{total_mode}_merge{merge_not_passed}.csv")
        data_from_file_treshold_passed.to_csv(dest_location, index=False)
        return data_from_file_treshold_passed
        
        