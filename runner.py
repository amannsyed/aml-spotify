from modules import convert_csv
from modules import dataClass
from modules import plot_and_extract_data
from modules import ml_models
import os
import numpy as np

def main():
    
    #src_data = os.getcwd() +os.sep+ "datasets" +os.sep+ "Spotify_Dataset_V3.csv"
    #dst_data = os.getcwd() +os.sep+ "datasets" +os.sep+ "excel_friendly_data.csv"
    #convert_csv.change_delimiter_in_a_file(src_data, dst_data)
    #
    #plotter_Obj = plot_and_extract_data.plot_and_extract_data()
#
    #features = ["Artist (Ind.)", "Nationality", "Continent"]
    #for feature in features:
    #    plotter_Obj.points_per_author_type_per_year(feature)
    #
    #plotter_Obj.plot_features_over_time(mode = "all", scaling= "minmax")
    #plotter_Obj.plot_features_over_time(mode = "all")
    #
    #plotter_Obj.plot_mean_vals_of_features_over_ranks(size_of_groups=10)
    #for year in range(2017, 2023+1):
    #    plotter_Obj.plot_mean_vals_of_features_over_ranks(year=year, size_of_groups=10)
    #
    #
    #for feature in features:
    #    plotter_Obj.points_per_author_type_per_month(feature)  
    #
    #extracted_data_loc = os.getcwd() +os.sep+ "extracted_data" +os.sep+ "points"
    #for file in features:
    #    plotter_Obj.convert_points_in_a_file_to_percentages(extracted_data_loc +os.sep+ f"{file}.csv")
    #    plotter_Obj.convert_points_in_a_file_to_percentages(extracted_data_loc +os.sep+ f"{file}_monthly.csv")
    
    
    #file = os.getcwd() +os.sep+ "extracted_data" +os.sep+ "points" +os.sep+ "Nationality_prercentage.csv"
    #plotter_Obj.drop_authors_with_low_percentage(source_file = file, total_mode= False, threshold= 0.01, merge_not_passed= True)
    #plotter_Obj.drop_authors_with_low_percentage(source_file = file, total_mode= True, threshold= 0.01, merge_not_passed= True)
     
    ###################################################################################### 
    ###################################################################################### 
    ######################################################################################
    #                    comment out one of the lines below:
    ######################################################################################
    #ml_models.dummyPredictor(label_type = "song_count", binary = True, threshold = 135)
<<<<<<< Updated upstream
    ml_models.find_best_RDF(label_type = "song_count", binary = True, threshold = 135)
    ml_models.find_best_SVM(label_type = "song_count", binary = True, threshold = 135)
    #ml_models.find_best_MLP(label_type = "song_count", binary = True, threshold = 135)
    #ml_models.find_best_TF_FFN(label_type = "song_count", binary = True, threshold = 135)
=======
    # ml_models.find_best_RDF(label_type = "song_count", binary = True, threshold = 135)
    #ml_models.find_best_SVM(label_type = "song_count", binary = True, threshold = 135)
    # ml_models.find_best_MLP(label_type = "song_count", binary = True, threshold = 135)
    ml_models.find_best_TF_FFN(label_type = "song_count", binary = True, threshold = 135)
>>>>>>> Stashed changes
    ######################################################################################
    ######################################################################################
    ######################################################################################
    
    
    
if __name__ == "__main__":
    main()