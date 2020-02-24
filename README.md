# IntrusionDetectionSystem
Autoencoder based intrusion detection system trained and tested with the CICIDS2017 data set.  

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

One of my initial intuitions, that turned out wrong uppn testing, was that for the packet length fields the standard deviation and mean would be sufficient.  My guess was that many network attacks would send nearly identical packets repeatedly (DDoS attacks, port scanning, brute force attacks).  Testing seemed to disprove this and I quickly expanded the feature set to include both minimum and maximum packet length as well as the statistical measures.  In hindsight, I should have analyzed which attacks were successfully detected and which were not to see if my intuition was at least correct for the subset of attacks that drove my thinking.
