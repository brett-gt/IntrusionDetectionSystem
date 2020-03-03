# IntrusionDetectionSystem
Autoencoder based intrusion detection system trained and tested with the CICIDS2017 data set.  Currently implemented using Python and Tensorflow 2.0.

## Current Best Network

The current best network uses a two-layer sparse autoencoder with L1 kernel regularization on the hidden layer.  On data it has previously not seen, this network is capable of correctly identifying 91.3% of BENIGN data (8.7% false positive rate) and 98.3% of attacks (1.7% false negative rate).  

The ability of the network (trainined only on Day 1 BENIGN data) to detect specific attacks it had never seen before is shown below:  

|Attack	|Network Decision  | Count 	|
|-------|:-----:|:--------:|
|BENIGN	|FALSE	| 2075854
|	      |TRUE  |	197243
|Bot	  |FALSE	| 1677
|	      |TRUE | 	289
|DDoS	  |TRUE	|  128027
|DoS GoldenEye	|TRUE	 |  10293
|DoS Hulk	| TRUE  |	231073
|DoS Slowhttptest |	TRUE	|  5499
|DoS slowloris |	TRUE	|  5276
|	       | FALSE	|  520
|FTP-Predator	|TRUE	|  3974
|	       |FALSE	 |  3964
|Heartbleed |	TRUE	|  11
|Infiltration |	TRUE |	29
|	       | FALSE	|  7
|PortScan	|TRUE	|  158930
|SSH-Patator |	TRUE	|  2979
|	           |FALSE	|  2918
|Web Attack Brute Force	| TRUE	|  1507
|Web Attack SQL Injection |	TRUE	 | 21
|Web Attack XSS	|  TRUE   |	652


The results can be reproduced by first downloading the CICIDS2017 dataset and pre-processing it using the encoding that is hard-coded in the DataEncoding.py file:

```python
path = WHEREVER_YOU_PUT_THE_CSV_FILES
encoding = DataE.AUTOENCODER_PREPROCESS
PreProc.process_folder(path, encoding)
```

A network can then be trained and tested using the following commands:
```python
pickles_path = "Pickles/New/"
params = HyperP.cAutoHyper("test.h5",[65, 64],[0.005, 0], [0.1, 0.3, 0])
encoding = DataE.AUTOENCODER_PREPROCESS

train_file = "Monday-WorkingHours.pkl"
train_data = pd.read_pickle(pickles_path + train_file)
model = IDS.train(train_data, encoding, params)

test_data = pd.read_pickle(pickles_path + "encode_test.pkl") 
thresh, score = IDS.find_best_thresh(model, encoding, test_data, 50)
```

## Informal Findings
The following is an informal list of ideas that were tested during various stages of development.  These informal test guided my attempts to construct an optimal solution but were not exhaustively tested.

1. A two-stage autoencoder (input layer -> hidden encoder layer -> bottleneck -> hidden decoder layer -> output layer) provided better results than a three-stage autoencoder (two hidden encoder and two hidden decoder layers).

2. Experimented with both L1 and L2 kernel and activity regularizers.  L2 kernel regularization showed improved results with a fully-connected network.  L1 activity regularized on just the hidden layer seemed to improve the sparse network. 

3. Exerimented with fully connected and a sparse autoencoder architecture.  Sparsity applied to the encoding layer (with more neurons to start with) produced better results.

4. Tried quantile normalization.  Produced inferior results.  


## TODO List:
The below list contains list of ideas to be tested:

1. Tune across multiple random seeds.  All testing thus far used a fixed random seed which will affect results, especially when performing fine-grained tuning of hyperparameters.

2. Selectively implement quantile normalization on appropriate values.

3. Do a broad grid search for ideal hyperparamaters using a quantile normalized data set.  Initial testing applied quantile normalized data to a specific set of hyperparameter that had been tuned using minmax normalized data.

4. Test without the special encoding of source/destination IP address to see if this approach is adding anything of value.

5. Expand grid search to include activation functions.

6.  Investigate applying contraints (regularization, drop out) on the decoding layer.  Initial testing seemed to indicate better results when only applying to encoding layer, but did not exhaustively test.  

7.  Perform more comprehensive tuning with both kernel and activity regularizers.  

8.  Train/test on all features in the CICIDS2017 data set instead of the subset I somewhat arbitrarily arrived at.  



## Overview
An autoencoder will be constructed and trained to detect network anomalies.  The goal with the autoencoder is to perform dimensionality reduction on the input variables to identify features unique to normal network data.  When abnormal network data is applied to the autoencoder, the network output will show poor correlation with the input data.  Looking for these areas of poor correlation allow the system to separate between normal and abnormal data.

To implement this approach, the autoencoder is trained on a set of data which contains no anomalies to solve for the identity function.  A loss function, such as mean absolute error (MAE), is applied to identify the correlation between the network inputs and outputs.  By minimizing this loss function across a wide variety of normal network data, the hope is for abnormal test data to show excessive error.  Test data can then be classified using a threshold error value to separate normal from abnormal data.

## Data Set
The Intrusion Detection System operates on the [CICIDS2017] (https://www.unb.ca/cic/datasets/ids-2017.html) data set provided by the Canadian Institute of Cybersecurity (CIC). The data set represents five days worth of traffic (Monday through Friday) with Monday’s data containing no network intrusions.  Tuesday through Friday contains a mix of good and intrusion data.  The ratio of good to intrusion packets vary based on type of intrusion (i.e. a DDOS attack has many more intrusion packets than a SQL Injection attack).

The data set is available both as raw PCAP captures and as a post-processed data set generated from their [CICFlowMeter] (http://www.netflowmeter.ca/netflowmeter.html) tool. The post-processed data groups raw packets into data flows and provides 83 statistical features (counts, minimum, maximum, mean, standard deviation) of values like number of packets, packet sizes, data rates, IP falgs, and duration.  To simplify implementation and run-time, I trained and tested with the post-process data provided by the CIC using the CICFlowMeter tool.  For a real-world application, an architecture could be created that used the CICFlowMeter in real-time to feed the IDS.  

An interesting future project would be to implement a network on the raw data.  This would probably require some sort of LSTM autoencoder or an LSTM front-end to an autoencoder since the raw data would appear as time series data.

## Data Preprocessing
Pre-processing of the data set is a critical step part of the solution which can have significant impacts on the results.  Data pre-processing includes steps like feature selection, feature encoding/normalization, and scaling of data.  Any erroneous data (infinite value, NaN, etc) is also removed.  Pre-processing of data can also be used to augment the raw network traffic with known configuration data (i.e. is an IP address internal or external to the network, are certain ports dedicated to known applications in the network, etc).

### Feature Selection
Feature selection is used to reduce the dimensionality of the data prior to use (i.e. drop some of the columns in the data set).  Two major reasons to perform feature selection are: the feature does not provide any value and the feature is not orthogonal with other features (i.e. the information it provides is already found in other features).  For feature selection, I used a combination of intuition (which in my case is probably as likely to be hurt as help), feature analysis algorithms like Principle Component Analysis, and research from published IDS papers. Specifically, the The most interesting of those papers was: https://www.mdpi.com/2079-9292/8/3/322/pdf

One of my initial intuitions, that turned out wrong, was that for the packet length fields the standard deviation and mean would be sufficient.  My guess was that many network attacks would send nearly identical packets repeatedly (DDoS attacks, port scanning, brute force attacks).  Testing seemed to disprove this and I quickly expanded the feature set to include both minimum and maximum packet length as well as the statistical measures.  In hindsight, I should have analyzed which attacks were successfully detected and which were not to see if my intuition was at least correct for the subset of attacks that drove my thinking. 

### Normalization
Most of the values in the CICIDS data set are numerical and describe things like packet lengths, data rates, and counts of the number of times various flags occurred.  Normalizing numerical values to a range that matches the activation functions used in the numerical network (usually 0 to 1 or -1 to 1) is common practice and has been demonstrated to improve network performance.  

Most testing (and results recorded here) was performed using minmax normalization on all numerical values.  

In analysis of the data set, I did notice that many numeric values have a “barbell distribution” where a majority of the values are either very small or very large.  For example, most flow durations are either very short (<500 us) or very long (>10,000,000+ us).  Because most of the data is bunched at the extremes, I thought there could be value in spreading the values out more evenly. To do this, I used quantile normalization to to create a more uniform distribution.  Turns out I was wrong.  When applying quantile normalized data to the baseline network, a substantial performance degradation was noted.  

Some caveats realting to quantile distribution that I haven’t tested yet:  

1.	Quantile normalization was applied to all numeric values even though some do not demonstrate the barbell distribution.  More selective application of the quantile normalization could yield better results.

2.	It could be possible that a different network configuration combined with quantile normalization yields better results.  

### Encoding

Categorical values in the CICIDS data set include values such as IP addresses, ports, and protocols.  

The destination port is an important discriminator in network data because many protocols have well-defined destination ports.  For example, SSH protocol typically uses destination port 22, Domain Name Service typical uses destination port 53, and Hypertext Transfer Protocol typical uses destination port 88.  Even when they aren’t universally recognized, it is common for vendor unique software to use common port definitions.  

The difficulty in encoding destination port is that IP communication allows 65535 unique port numbers.  One-hot encoding port number would massively grow the number of input neurons required.  So while I wanted to treat destination port as a categorical value, I ended up applying minmax normalization to it.

Many machine learning IDS systems in literature discard source and destination IP addresses.  However, I created a special encoding method for these values.  The network architecture is defined in the CICIDS2017 data set such that three classes of IP addresses can be identified: internal to the network, external to the network, and gateway IP addresses (those connected to both internal and external networks).  In a real-world IDS application, these three classes of IP addresses would also be known.  Using this knowledge, the source and destination IP addresses were one-hot encoded as one of those three three classes.  As a result, six one-hot encoded features (Internal Source IP, Internal Dest IP, Gateway Source IP, Gateway Dest IP, External Source IP, and External Dest IP) were added to the data set.

In the CICIDS2017 data set, the protocol field was an integer value describing the protocol with only three values being present (0 = HOPOPT?, 6 =TCP, 17 = UDP).  In encoding, I one-hot encoded the protocol field.   


## Autoencoder Architecture

The general approach I used to create the optimum autoencoder architecture was to start with a basic autoencoder and perform grid searches across various hyperparameters.  All autoencoder architectures used were the typical layout where there is an encoding layer feeding into a bottleneck and then expanding back out in a decoding layer.  From the perspective of number of neurons, the encoding and decoding layers are mirror images of each other.  Other hyperparameters, such as sparsity and regularization, have primarily been applied only to the encoding layer.  Based on limited experimentation, only applying these hyperparameters to the encoder seemed to produce better results.  A cursory look through literature did not seem to indicate if these parameters should be applied to both the encoder and decoder layers.  

To provide reproducible results, all random value seeds were fixed as “1234”.  Testing was performed to demonstrate randomization was removed from the network such that runs with the same hyperparameters resulted in identical networks and results.  There is a possibility that this arbitrary random value seed has affected results described later, especially when since some tuning results don’t appear to be clear minimums/maximums.  On my list of things to do is to tune across multiple random seeds.  


## Threshold

The loss function currently used is the mean absolute error (MAE) between the decoded result and the original input.  Once trained, a threshold MAE is established which creates a decision between ‘normal’ and ‘anomalous’ data.  The best threshold possible was solved for empirically by determining which threshold provides the maximum true positive anomalies and true negative normal values.  





