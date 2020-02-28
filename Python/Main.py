import numpy as np
import tensorflow as tf
import pandas as pd
import os
import IDS
import Preprocess as PreProc

import Hyper as HyperP
import DataEncoding as DataE

#-------------------------------------------------------------------------------------
def main():   
    pickles_path = "Pickles/New/"
    params = HyperP.cAutoHyper("test.h5",[65, 64],[0.005, 0], [0.1, 0.3, 0])
    encoding = DataE.AUTOENCODER_PREPROCESS

    print("Loading training data...")
    train_file = "Monday-WorkingHours.pkl"
    train_data = pd.read_pickle(pickles_path + train_file)

    print("Training...")
    model = IDS.train(train_data, encoding, params)
    model.save("trained.h5")

    print("Loading test data...")
    test_data = pd.read_pickle(pickles_path + "encode_test.pkl") 

    print("Finding best thresh...")
    thresh, score = IDS.find_best_thresh(model, encoding, test_data, 50)


#-------------------------------------------------------------------------------------
def findThresh():
    pickles_path = "Pickles/New/"
    encoding = DataE.AUTOENCODER_PREPROCESS

    print("Testing...")
    test_data = pd.read_pickle(pickles_path + "Friday-WorkingHours-Afternoon-DDos.pkl") 
    model = tf.keras.models.load_model("new_encoding.h5")
    IDS.find_best_thresh(model, encoding, test_data, 50)

#-------------------------------------------------------------------------------------
def trainNetwork():
    print("Training...")
    pickles_path = "Pickles/New/"
    train_file = "Monday-WorkingHours.pkl"

    train_data = pd.read_pickle(pickles_path + train_file)
    params = HyperP.cAutoHyper("test.h5",[65, 64],[0.005, 0], [0.1, 0.3, 0])
    encoding = DataE.AUTOENCODER_PREPROCESS

    model = IDS.train(train_data, encoding, params)
    model.save("new_encoding.h5")

#-------------------------------------------------------------------------------------
def generateDataSet():
    path = "Data/CIC2017/GenLabels/" 
    pickles_path = "Pickles/New/"
    file = "Monday-WorkingHours.pcap_ISCX.csv" 

    path = WHEREVER_YOU_PUT_THE_CSV_FILES
    encoding = DataE.AUTOENCODER_PREPROCESS
    PreProc.process_folder(path, encoding)


#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
