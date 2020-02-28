""" Preprocess contains functions used to preprocess dataset including feature selection and
    normalization.  Specific details of operations to be performed are defined in Settings.py.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
import os

#--------------------------------------------------------------------------------
def process_folder(path, data_encoding, preamble=""):
    """Loop through all CSVs in a given folder and preprocess/save each as pickle.
    """
    files = os.listdir(path)
    files_no_ext = [x.split('.')[0] for x in files]
    files_pkl = [x + ".pkl" for x in files_no_ext]
  
    for file in files:
        if ~os.path.isdir(path): 
            print("\n\n\nProcessing " + file)
            process_file(path, preamble+file, data_encoding)

    print("Creating combined test pickle...")
    combined = pd.concat([pd.read_pickle(fp) for fp in files_pkl], ignore_index=True)
    combined.to_pickle(data_encoding.FILENAME)


#################################################################################
# THOUGHT: Normalization may be differnet on differnet days of data (for example if the max
# value in that data set changes.  Could do some custom normalization techniques that carry over
# the training set max or guess at an absolute max for normalization.
# https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
#--------------------------------------------------------------------------------
def process_file(path, filename, data_encoding):
    """Function to read CSV of the CICIDS2017, pre-process, and save as a pickle.
       
       Intent is for mot of the operation of this function to be controlled through the
       values in Settings.py.  There are a columns that are hardcoded in this function:
       manipulating IP addresses, the label, etc.

       Arguments
        path -> path of the source file
        filename -> filename of the csv file being parsed (also the output file name)
        data_encoding -> instance of DataEncoding class defining feature selection/extraction 
    """
    # TODO: Should probably have this return clean_data and leave file saving to another function
    
    output_filename = filename.split('.')[0] + ".pkl"

    #Load the data
    print("Reading " + filename)
    data = pd.read_csv(path + filename, na_values = 'Infinity', encoding = 'unicode_escape') # na_values = 'Infinity' because not standard Panda term

    print("Fixing column names...")
    data.columns = [a.lstrip() for a in data.columns] #Columns have leading space for some reason

    data['Full Label'] = data['Label']  #Adding a copy column which doesn't appear in original because going to replace label later

    #Create a second set that removes columns we don't want.  
    print("Creating desired subset of data...")
    clean_data = data[data_encoding.COL_TO_USE]
    print(clean_data.head())

    #TODO Could also consider removing these rows instead of replacing values
    print("Replacing NaN with column mean.")
    clean_data.select_dtypes(include=[np.float64, np.int64]).apply(lambda x: x.fillna(x.mean()),axis=0)
    
   
    print("Scaling data columns...")
    scaler = data_encoding.NORM_METHOD
    clean_data[data_encoding.COL_TO_NORM] = scaler.fit_transform(clean_data[data_encoding.COL_TO_NORM])

 
    print("One-hot encoding data columns...")
    for x in data_encoding.COL_TO_ONEHOT:
        print("Performing one-hot encoding on " + x + " which has " + str(data[x].nunique()) + " unique values.")
        dummies = pd.get_dummies(data[x],prefix=x)
        clean_data.drop(x, axis=1, inplace=True)                   #Remove old column
        clean_data = pd.concat([clean_data,dummies],axis=1)        #Add new columns
        #print(clean_data.head())

    print("Including columns that don't need formatting.")
    clean_data = pd.concat([clean_data, data[data_encoding.COL_NO_FORMAT]],axis=1)
                           
    print(clean_data['Label'].unique())

    print("Replacing labels with dictionary mapping...")
    clean_data['Full Label'] = clean_data['Full Label']
    clean_data['Label'] = clean_data['Label'].map(data_encoding.LABEL_MAPPING)

    print("Inserting IP addresses encoding...")
    clean_data['Internal Source IP'] = data['Source IP'].isin(data_encoding.INTERNAL_NETWORK_IPs).astype(int)
    clean_data['Internal Dest IP'] = data['Destination IP'].isin(data_encoding.INTERNAL_NETWORK_IPs).astype(int)
    clean_data['Public Facing Source IP'] = data['Source IP'].isin(data_encoding.PUBLIC_FACING_IPs).astype(int)
    clean_data['Public Facing Dest IP'] = data['Destination IP'].isin(data_encoding.PUBLIC_FACING_IPs).astype(int)
    clean_data['External Source IP'] = (~data['Source IP'].isin(data_encoding.INTERNAL_NETWORK_IPs + data_encoding.PUBLIC_FACING_IPs)).astype(int)
    clean_data['External Dest IP'] = (~data['Destination IP'].isin(data_encoding.INTERNAL_NETWORK_IPs + data_encoding.PUBLIC_FACING_IPs)).astype(int)
        
    print("Done creating data... statistics to follow:")
    print(clean_data.describe(include='all'))

    print("Writing to pickle: " + output_filename)
    clean_data.to_pickle(output_filename)


#--------------------------------------------------------------------------------
def normalization_test(path, file):
    """Function to look at effects of different types of normalization.

       Currently just prints out statistics.  Could do some plotting or other 
       visualizations to help provide more detail.

       Arguments
        path -> path of the source file
        file -> file name of the csv file being parsed (also the output file name)
    """
    # TODO: Many of the data types have a barbell distribution (bunch of very small
    # and very big values.  Try to find a normalization method that helps deal with that.
    print("Normalization Test")
    print("\nReading " + file)
    data = pd.read_csv(path + file, na_values = 'Infinity', encoding = 'unicode_escape') # na_values = 'Infinity' because not standard Panda term
    subframe = data[[' Flow Duration',' Total Fwd Packets']]

    print("\n\nRaw data:")
    print(subframe.describe(include='all'))

    print("\n\nMinMax")
    scaler = preprocessing.MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(subframe), 
                               columns=subframe.columns, 
                               index=subframe.index)
    print(scaled_data.describe(include='all'))

    print("\n\nPower")
    scaler = preprocessing.PowerTransformer()

    scaled_data = pd.DataFrame(scaler.fit_transform(subframe), 
                               columns=subframe.columns, 
                               index=subframe.index)
    print(scaled_data.describe(include='all')) #was just scaled_data.descirbe

    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html
    print("\n\nQuantile")
    scaler = preprocessing.QuantileTransformer()
    scaled_data = pd.DataFrame(scaler.fit_transform(subframe),
                               columns=subframe.columns, 
                               index=subframe.index)
    print(scaled_data.describe(include='all')) #was just scaled_data.descirbe


    #Tansig normalization - tansig(n) = -1 + 2/(1+e^-2n)


#--------------------------------------------------------------------------------
def analyze(data):
    """Helper function to look at each column of data and provide basic statistics.
    """
    columns = list(data)
    print(columns)
    for i in columns:
        print(i)
        print(i + " is data type " + str(data[i].dtype) + " with " + str(data[i].nunique()) + " unique values.")

        if data[i].dtype == 'float64':
            nan_indices = np.argwhere(np.isnan(data[i].values))
            print('#{} nan elements'.format(nan_indices))

            inf_indices = np.argwhere(np.isinf(data[i].values))
            print('#{} inf elements'.format(inf_indices))


#--------------------------------------------------------------------------------
def one_hot_column(df, name):
    """Performs one-hot encoding of a column of a dataframe.

    Function removes the original column from the data frame and replaces it with
    the one-hot encoded functions.

    Arguments:
        df -> dataframe to be operated on
        name -> name of the column to be one-hot encoded, also used as part of the
                new name of the one-hot enncoded columns.
    """
    dummies = pd.get_dummies(df[name])
    print(dummies.head())
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)





    