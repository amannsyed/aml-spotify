import pandas as pd# Imports from __future__ since we're running Python 2
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.pipeline import Pipeline
from itertools import product



def get_X_and_Y():
    #read csv file 
    file_path = 'your_file.csv'
    df = pd.read_csv(file_path)

    # using HOLD-OUT validation method 
    X = df[['Danceability','Energy','Loudness','Speechiness','Acousticness','Instrumentalness','Valence']]
    threshold = 500
    df['binarized_column'] = df['song_count'].apply(lambda x: 1 if x > threshold else 0)
    y= df['binarized_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=0)


    return  X_train, y_train, X_test, y_test

def Multinomial_Naive_Bayes_classifier():
    X_train, y_train, X_test, y_test= self.get_X_and_Y()

    #train a Multinomial Naive Bayes classifier
    mnb = MultinomialNB()
    mnb.fit(X=X_train, y=y_train)

    #checking accuracy score
    tr_pred = mnb.predict(X=X_train)
    ca = accuracy_score(y_train, mnb.predict(X_train))

    # confusion matrix 
    cm = confusion_matrix(y_train, tr_pred)
    cm

    #Normalising the produced confusion matrix by the true class and display the result.
    cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
    cm_norm 


def  Gaussian_Naive_Bayes_classification():

    X_train,y_train, X_test, y_test = self.get_X_and_Y()

    X_train = news_A_clean.drop('class',axis=1)
    y_train = news_A_clean['class']

    clf = GaussianNB()

    clf.fit(X_train,y_train)
    ca = clf.score(X_train,y_train)

    print("The classification accuracy is {}".format(ca))

    #Plot the (normalised) confusion matrix for the training data.
    cm = sklearn.metrics.confusion_matrix(y_train,clf.predict(X_train))
    cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        
        
def get_X_and_Y_v2(label_type:str, binary:bool = False, threshold:int = 0, Data_for_dummy = False):
    PLOTS_PATH = os.path.join(os.getcwd(), 'plots')
    EXTRACTED_DATA_PATH =os.path.join(os.getcwd(), "extracted_data")
    file_with_occurences = os.path.join(EXTRACTED_DATA_PATH, "data_for_ML" +os.sep+ "data_6_all_all.csv")
    
    extracted_data = pd.read_csv(file_with_occurences, delimiter = ',')
    
    
    if Data_for_dummy:
        X = extracted_data[['Artists', 'Nationality']]
    # using HOLD-OUT validation method 
    else:
        columns_X = ['Danceability','Energy','Loudness','Speechiness','Acousticness','Instrumentalness','Valence', 'artists_points', 'artists_songs_counted', 'nationality_points', 'nationality_songs_counted']
        X = extracted_data[columns_X].copy(deep=True)
        scaler = MinMaxScaler()
        for category in columns_X:
            X[category] = scaler.fit_transform(X[[category]])
    if binary:
        extracted_data['binarized_column'] = extracted_data[label_type].apply(lambda x: 1 if x > threshold else 0)
        y= extracted_data['binarized_column']
    else:
        y= extracted_data[label_type]

    
    #extracted_data.to_csv("test.csv", sep=',', encoding='utf-8', index=False, header=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=0)

    return  X_train, y_train, X_test, y_test
        
    
def randomForest_model(label_type:str, binary:bool = False, threshold:int = 0, model_args:dict= None):
    
    max_depth =    model_args['max_depth']
    n_estimators = model_args['n_estimators']
    balace_type =  model_args['balace_type']
    
    X_train,y_train, X_test, y_test = get_X_and_Y_v2(label_type = label_type, binary = binary, threshold = threshold)
    negative_class_ratio = np.mean(y_train == 0)*100
    positive_class_ratio = 100- negative_class_ratio
    print(f"negative_class_ratio = {negative_class_ratio}")
    print(f"positive_class_ratio = {positive_class_ratio}")
    
    weight_dict = {1: negative_class_ratio, 0: positive_class_ratio}
    
    if balace_type == "weight_dict":
        class_weight = {1: negative_class_ratio, 0: positive_class_ratio}
    elif "balanced_subsample":
        class_weight = "balanced_subsample"
    else:
        class_weight = "balanced"
    
    rdf = RandomForestClassifier(criterion = "entropy", max_depth= max_depth, random_state= 1000, n_estimators=n_estimators, class_weight=class_weight)
    #rdf = RandomForestClassifier(criterion = "entropy", max_depth= 100, random_state= 1000, n_estimators=500)
    
    #ros = RandomOverSampler(random_state=42)
    #X_train, y_train = ros.fit_resample(X_train, y_train)
    
    
    #return
    rdf.fit(X_train, y_train)
    rdf.score(X_test, y_test)

    Conf_matrix = confusion_matrix(y_test, rdf.predict(X_test))
    print(f"RDF Conf_matrix = \n{Conf_matrix}")
    #Conf_matrix = Conf_matrix / (Conf_matrix[1][0]+Conf_matrix[1][1])
    #print(f"normalised Conf_matrix = {Conf_matrix}")
    #plot_confusion_matrix(Conf_matrix)
    del(rdf)
    return Conf_matrix


def MLP_model(label_type:str, binary:bool = False, threshold:int = 0, model_args:dict= None):
    
    hidden_layer_sizes = model_args["hidden_layer_sizes"]
    activation = model_args["activation"]
    
    X_train,y_train, X_test, y_test = get_X_and_Y_v2(label_type = label_type, binary = binary, threshold = threshold)
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=1000)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

    Conf_matrix = confusion_matrix(y_test, model.predict(X_test))
    print(f"MLP Conf_matrix = \n{Conf_matrix}")
    del(model)
    return Conf_matrix



def svm_model(label_type:str, binary:bool = False, threshold:int = 0, model_args:dict= None):
    
    
    kernel= model_args["kernel"]
    C = model_args["C"]
    balace_type = model_args["class_weight"]
    probability = model_args["probability"]
    
    X_train,y_train, X_test, y_test = get_X_and_Y_v2(label_type = label_type, binary = binary, threshold = threshold)
    
    negative_class_ratio = np.mean(y_train == 0)*100
    positive_class_ratio = 100- negative_class_ratio
    print(f"negative_class_ratio = {negative_class_ratio}")
    print(f"positive_class_ratio = {positive_class_ratio}")
    
    weight_dict = {1: negative_class_ratio, 0: positive_class_ratio}

    if balace_type == "balanced":
        class_weight = "balanced"
    elif "dict":
        class_weight = weight_dict
    else:
        class_weight = None
    
    model = svm.SVC(kernel=kernel, C=C, class_weight = class_weight, probability = probability)
    #model = svm.SVC(kernel='sigmoid', C=1, class_weight = "balanced")

    
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

    Conf_matrix = confusion_matrix(y_test, model.predict(X_test))
    print(f"SVM Conf_matrix = \n{Conf_matrix}")
    del(model)
    return Conf_matrix


def tensorflowFFN(label_type:str, binary:bool = False, threshold:int = 0, model_args:dict= None):
    
    input_layer_neurons = model_args["input_layer_neurons"]
    input_layer_activation = model_args["input_layer_activation"]
    hidden_layers_neurons = model_args["hidden_layer_neurons"]
    hidden_layers_activations = model_args["hidden_layer_activations"]
    weight = model_args["weight"]
    
    
    X_train,y_train, X_test, y_test = get_X_and_Y_v2(label_type = label_type, binary = binary, threshold = threshold)
    
    #ros = RandomOverSampler(random_state=42)
    #X_train, y_train = ros.fit_resample(X_train, y_train)
    #pipeline = Pipeline([
    #('smote', SMOTE()),
    #('under_sampler', RandomUnderSampler())
    #])
    #
    negative_class_ratio = np.mean(y_train == 0)
    positive_class_ratio = 1- negative_class_ratio
    print(f"negative_class_ratio = {negative_class_ratio}")
    print(f"positive_class_ratio = {positive_class_ratio}")
    
    weight_dict = {1: negative_class_ratio, 0: positive_class_ratio}

    #X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    
    
    model = Sequential()

    # Input layer
    model.add(Dense(input_layer_neurons, input_shape=(X_train.shape[1],), activation=input_layer_activation))
    model.add(tf.keras.layers.Dropout(0.4))
    # Hidden layers
    for i, hidden_layer_neurons in enumerate(hidden_layers_neurons):
        model.add(Dense(hidden_layer_neurons, activation=hidden_layers_activations[i]))
        model.add(tf.keras.layers.Dropout(0.4))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification, using sigmoid activation

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics="accuracy")

    # Print a summary of the model's architecture
    model.summary()
    if weight == "dict":   
        model.fit(X_train, y_train, epochs=25, class_weight=weight_dict)
    else:
        model.fit(X_train, y_train, epochs=25)
        
    #model.fit(X_resampled, y_resampled, epochs=200)
    
    y_pred_prob = model.predict(X_test)
    # Convert probabilities to binary predictions using the specified threshold
    #for thr in range(0, 11):
    #    y_pred_binary = (y_pred_prob > (float(thr)/10)).astype(int)
#
    #    Conf_matrix = confusion_matrix(y_test, y_pred_binary)
    #    print(f"tensorflow FFN Conf_matrix boundary ({float(thr)/10}) = \n{Conf_matrix}\n\n")

    y_pred_binary = (y_pred_prob > 0.5).astype(int)
    Conf_matrix = confusion_matrix(y_test, y_pred_binary)
    print(f"tensorflow FFN Conf_matrix boundary (0.5) = \n{Conf_matrix}\n\n")
    del(model)
    return Conf_matrix


def good_artis_or_nationality_is_there(inputX, goodAtrists, goodNationalities):
    
    
    if goodAtrists is not None:
        artists_list = str(inputX["Artists"])
        artists_list= artists_list.replace("]", "")
        artists_list= artists_list.replace("[", "")
        artists_list= artists_list.replace("'", "")
    
        for artist in str(artists_list).split(", "):
            
            if artist in list(goodAtrists):
            #exit(1)
                return 1
            
    if goodNationalities is not None:
        nationalities_list = str(inputX["Nationality"])
        nationalities_list= nationalities_list.replace("]", "")
        nationalities_list= nationalities_list.replace("[", "")
        nationalities_list= nationalities_list.replace("'", "")
    
        for nationality in nationalities_list.split(", "):
        #print(artist)
            if nationality in list(goodNationalities):
            #exit(1)
                return 1
    #exit(1)
    
    
    return 0
        

def dummyPredictor(label_type:str, binary:bool = False, threshold:int = 0):
    X_train,y_train, X_test, y_test = get_X_and_Y_v2(label_type = label_type, binary = binary, threshold = threshold, Data_for_dummy= True)
    
    PLOTS_PATH = os.path.join(os.getcwd(), 'plots')
    EXTRACTED_DATA_PATH =os.path.join(os.getcwd(), "extracted_data")
    file_with_artists = os.path.join(EXTRACTED_DATA_PATH, "points" +os.sep+ "Artist (Ind.)_prercentage.csv")
    file_with_nationalities = os.path.join(EXTRACTED_DATA_PATH, "points" +os.sep+ "Nationality_prercentage.csv")
    
    good_artists = pd.read_csv(file_with_artists, delimiter = ',')
    num_of_all_artists = good_artists["author"].shape[0]
    good_artists_data = good_artists["author"].head(int(0.1*num_of_all_artists)).copy(deep = True)
    
    good_nationalities = pd.read_csv(file_with_nationalities, delimiter = ',')
    num_of_all_nationalities = good_nationalities["author"].shape[0]
    good_nationalities_data = good_nationalities["author"].head(int(0.1*num_of_all_nationalities)).copy(deep = True)
    
    
    dummy_args1 = [good_artists_data, good_nationalities_data, "dummy_artists_and_nationalities.txt"]
    dummy_args2 = [good_artists_data, None, "dummy_artists_only.txt"]
    dummy_args3 = [None, good_nationalities_data, "dummy_nationalities_only.txt"]
    
    runDummyPredictions(X_test, y_test, dummy_args1[0], dummy_args1[1], dummy_args1[2])
    runDummyPredictions(X_test, y_test, dummy_args2[0], dummy_args2[1], dummy_args2[2])
    runDummyPredictions(X_test, y_test, dummy_args3[0], dummy_args3[1], dummy_args3[2])
    
    
def runDummyPredictions(X_test, y_test, good_artists_data, good_nationalities_data, file_name):
    best_models_path = os.path.join(os.getcwd(), "best_models")
    if not os.path.exists(best_models_path):
            os.mkdir(best_models_path)
    
    dummy_predictions = np.zeros(y_test.shape[0])
    counter = 0
    for row_id in y_test.index:
        dummy_predictions[counter] = 1 if good_artis_or_nationality_is_there(X_test.loc[row_id], good_artists_data, good_nationalities_data) else 0
        counter +=1 
    Conf_matrix_artists_only = confusion_matrix(y_test, dummy_predictions)
    print(f"dummy Conf_matrix = \n{Conf_matrix_artists_only}")
    artists_only_txt = f"Conf_matrix = \n{Conf_matrix_artists_only}\n\nCohensKappa = {getCohensKappa(Conf_matrix_artists_only)}\n\nF1 = {getF1Score(Conf_matrix_artists_only)}\n\n"
    with open(best_models_path +os.sep+ file_name, "w") as f:
        f.write(artists_only_txt)
            


def find_best_RDF(label_type:str, binary:bool = False, threshold:int = 0):
    best_models_path = os.path.join(os.getcwd(), "best_models")
    if not os.path.exists(best_models_path):
            os.mkdir(best_models_path)
    
    max_depth_list = [i for i in range(1,101, 10)]
    n_estimators_list = [i for i in range(1,101, 10)]
    balace_type_list = ["balanced", "weight_dict", "balanced_subsample"]
    
    list_of_all_combinations = product(max_depth_list, n_estimators_list, balace_type_list)

    dicts_of_all_combinations = [{"max_depth": el1, "n_estimators": el2, "balace_type": el3} for el1, el2, el3 in list_of_all_combinations]
    
    
    maxF1 = -999999999999
    maxCohensKappa = -99999999999
    model_with_best_F1 = ""
    model_with_best_CohensKappa = ""
    
    for model_args in dicts_of_all_combinations:
        print(f"hyperparameters = {model_args}")
        confM = randomForest_model(label_type = label_type, binary = binary, threshold = threshold, model_args= model_args)
        
        if maxCohensKappa < getCohensKappa(confM):
            model_with_best_CohensKappa = f"Conf_matrix = \n{confM}\n\nCohensKappa = {getCohensKappa(confM)}\n\nF1 = {getF1Score(confM)}\n\n" + f"hyperparameters = {model_args}"
            maxCohensKappa = getCohensKappa(confM)
        
        if maxF1 < getF1Score(confM):
            model_with_best_F1 = f"Conf_matrix = \n{confM}\n\nCohensKappa = {getCohensKappa(confM)}\n\nF1 = {getF1Score(confM)}\n\n" + f"hyperparameters = {model_args}"
            maxF1 = getF1Score(confM)
             
    
    with open(best_models_path +os.sep+ "best_RDF_F1.txt", "w") as f:
        f.write(model_with_best_F1)
    with open(best_models_path +os.sep+ "best_RDF_CohensKappa.txt", "w") as f:
        f.write(model_with_best_CohensKappa)
            
            
def find_best_SVM(label_type:str, binary:bool = False, threshold:int = 0):
    
    best_models_path = os.path.join(os.getcwd(), "best_models")
    if not os.path.exists(best_models_path):
            os.mkdir(best_models_path)
    
    kernel_list = ["sigmoid", "linear", "poly", "rbf"]
    C_list = [(1.0*i/10) for i in range(1, 21)]
    class_weight_list = ["balanced", "dict", "None"]
    probability_list = [True, False]
    
    list_of_all_combinations = product(kernel_list, C_list, class_weight_list, probability_list)

    dicts_of_all_combinations = [{"kernel": el1, "C": el2, "class_weight": el3, "probability": el4} for el1, el2, el3, el4 in list_of_all_combinations]
    
    
    maxF1 = -999999999999
    maxCohensKappa = -99999999999
    model_with_best_F1 = ""
    model_with_best_CohensKappa = ""
    
    for model_args in dicts_of_all_combinations:
        print(f"hyperparameters = {model_args}")
        confM = svm_model(label_type = label_type, binary = binary, threshold = threshold, model_args= model_args)
        
        if maxCohensKappa < getCohensKappa(confM):
            model_with_best_CohensKappa = f"Conf_matrix = \n{confM}\n\nCohensKappa = {getCohensKappa(confM)}\n\nF1 = {getF1Score(confM)}\n\n" + f"hyperparameters = {model_args}"
            maxCohensKappa = getCohensKappa(confM)
        
        if maxF1 < getF1Score(confM):
            model_with_best_F1 = f"Conf_matrix = \n{confM}\n\nCohensKappa = {getCohensKappa(confM)}\n\nF1 = {getF1Score(confM)}\n\n" + f"hyperparameters = {model_args}"
            maxF1 = getF1Score(confM)
             
    
    with open(best_models_path +os.sep+ "best_SVM_F1.txt", "w") as f:
        f.write(model_with_best_F1)
    with open(best_models_path +os.sep+ "best_SVM_CohensKappa.txt", "w") as f:
        f.write(model_with_best_CohensKappa)
        
        
def find_best_MLP(label_type:str, binary:bool = False, threshold:int = 0):
    
    best_models_path = os.path.join(os.getcwd(), "best_models")
    if not os.path.exists(best_models_path):
            os.mkdir(best_models_path)


    activation_list = ["identity", "logistic", "tanh", "relu"]
    
    num_of_neurons_in_one_layer = [i for i in range(1, 161, 20)]
    
    num_of_neurons_in_two_layers = list(product(num_of_neurons_in_one_layer, num_of_neurons_in_one_layer))
    num_of_neurons_in_two_layers = [list(i) for i in num_of_neurons_in_two_layers]
    
    num_of_neurons_in_three_layers = list(product(num_of_neurons_in_one_layer, num_of_neurons_in_one_layer, num_of_neurons_in_one_layer))
    num_of_neurons_in_three_layers = [list(i) for i in num_of_neurons_in_three_layers]
    
    num_of_neurons_in_one_layer= [[i] for i in num_of_neurons_in_one_layer]
    hidden_layer_sizes_list = []
    hidden_layer_sizes_list.extend([(i) for i in num_of_neurons_in_one_layer])
    hidden_layer_sizes_list.extend(num_of_neurons_in_two_layers)
    #hidden_layer_sizes_list.extend(num_of_neurons_in_three_layers)
    
    
    list_of_all_combinations = list(product(activation_list, hidden_layer_sizes_list))

    dicts_of_all_combinations = [{"activation": el1, "hidden_layer_sizes": tuple(el2)} for el1, el2 in list_of_all_combinations]
    
    
    maxF1 = -999999999999
    maxCohensKappa = -99999999999
    model_with_best_F1 = ""
    model_with_best_CohensKappa = ""
    
    for model_args in dicts_of_all_combinations:
        print(f"hyperparameters = {model_args}")
        confM = MLP_model(label_type = label_type, binary = binary, threshold = threshold, model_args= model_args)
        
        if maxCohensKappa < getCohensKappa(confM):
            model_with_best_CohensKappa = f"Conf_matrix = \n{confM}\n\nCohensKappa = {getCohensKappa(confM)}\n\nF1 = {getF1Score(confM)}\n\n" + f"hyperparameters = {model_args}"
            maxCohensKappa = getCohensKappa(confM)
        
        if maxF1 < getF1Score(confM):
            model_with_best_F1 = f"Conf_matrix = \n{confM}\n\nCohensKappa = {getCohensKappa(confM)}\n\nF1 = {getF1Score(confM)}\n\n" + f"hyperparameters = {model_args}"
            maxF1 = getF1Score(confM)
             
    
    with open(best_models_path +os.sep+ "best_MLP_F1.txt", "w") as f:
        f.write(model_with_best_F1)
    with open(best_models_path +os.sep+ "best_MLP_CohensKappa.txt", "w") as f:
        f.write(model_with_best_CohensKappa)
            
            
            
def find_best_TF_FFN(label_type:str, binary:bool = False, threshold:int = 0):
    
    best_models_path = os.path.join(os.getcwd(), "best_models")
    if not os.path.exists(best_models_path):
            os.mkdir(best_models_path)


    num_of_neurons_in_one_layer = [i for i in range(1, 161, 20)]
    activations_list = ["relu", "tanh", "sigmoid"]
    weight_list = ["dict", ""]
    
    num_of_neurons_in_one_layer = [i for i in range(1, 161, 20)]
    
    combinatios_no_hidden_layer = list(product(num_of_neurons_in_one_layer, activations_list, [[]], [[]], weight_list))
    combinatios_one_hidden_layer = list(product(num_of_neurons_in_one_layer, activations_list, [[i] for i in num_of_neurons_in_one_layer], [[i] for i in activations_list], weight_list))
    
    num_of_neurons_in_two_layers = list(product(num_of_neurons_in_one_layer, num_of_neurons_in_one_layer))
    num_of_neurons_in_two_layers = [list(i) for i in num_of_neurons_in_two_layers]
    activations_in_two_layers = list(product(activations_list, activations_list))
    activations_in_two_layers = [list(i) for i in activations_in_two_layers]
    
    combinatios_two_hidden_layers = list(product(num_of_neurons_in_one_layer, activations_list, num_of_neurons_in_two_layers, activations_in_two_layers, weight_list))
    
    
    num_of_neurons_in_one_layer= [[i] for i in num_of_neurons_in_one_layer]
    list_of_all_combinations = []
    list_of_all_combinations.extend(combinatios_no_hidden_layer)
    list_of_all_combinations.extend(combinatios_one_hidden_layer)
    #list_of_all_combinations.extend(combinatios_two_hidden_layers)
    
    
    dicts_of_all_combinations = [{"input_layer_neurons": el1, "input_layer_activation": el2, "hidden_layer_neurons": el3, "hidden_layer_activations": el4, "weight": el5} for el1, el2, el3, el4, el5 in list_of_all_combinations]
    
    
    maxF1 = -999999999999
    maxCohensKappa = -99999999999
    model_with_best_F1 = ""
    model_with_best_CohensKappa = ""
    
    for model_args in dicts_of_all_combinations:
        print(f"hyperparameters = {model_args}")
        confM = tensorflowFFN(label_type = label_type, binary = binary, threshold = threshold, model_args= model_args)
        
        if maxCohensKappa < getCohensKappa(confM):
            model_with_best_CohensKappa = f"Conf_matrix = \n{confM}\n\nCohensKappa = {getCohensKappa(confM)}\n\nF1 = {getF1Score(confM)}\n\n" + f"hyperparameters = {model_args}"
            maxCohensKappa = getCohensKappa(confM)
        
        if maxF1 < getF1Score(confM):
            model_with_best_F1 = f"Conf_matrix = \n{confM}\n\nCohensKappa = {getCohensKappa(confM)}\n\nF1 = {getF1Score(confM)}\n\n" + f"hyperparameters = {model_args}"
            maxF1 = getF1Score(confM)
             
    
    with open(best_models_path +os.sep+ "best_TF_FFN_F1.txt", "w") as f:
        f.write(model_with_best_F1)
    with open(best_models_path +os.sep+ "best_TF_FFN_CohensKappa.txt", "w") as f:
        f.write(model_with_best_CohensKappa)
     
            
def getF1Score(confMatrix):
    TP = confMatrix[1][1]
    TN = confMatrix[0][0]
    FP = confMatrix[0][1]
    FN = confMatrix[1][0]
    
    if (TP+FP) == 0 or (TP+FN) == 0:
        return 0
    
    Precision = (TP) / (TP+FP)
    Recall = (TP) / (TP+FN)
    
    return (2*Precision*Recall) / (Precision + Recall)

def getCohensKappa(confMatrix):
    TP = confMatrix[1][1]
    TN = confMatrix[0][0]
    FP = confMatrix[0][1]
    FN = confMatrix[1][0]
    
    total = TP+TN+FP+FN
    Po = (TP + TN) / total
    
    part1 = (TP + FP) / total
    part2 = (TP + FN) / total
    part3 = (TN + FN) / total
    part4 = (TN + FP) / total
    Pe = (part1*part2) + (part3*part4)
    
    return (Po - Pe)/(1 - Pe) 
    
    