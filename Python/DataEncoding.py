""" Holds configuration of data encoding paramters.
"""

from abc import ABCMeta, abstractmethod
from sklearn import preprocessing

#--------------------------------------------------------------------------------
# DATASET VALUES
#--------------------------------------------------------------------------------
# See http://www.netflowmeter.ca/netflowmeter.html for column definitions
#
# Flow ID	 Source IP	 Source Port	 Destination IP	 Destination Port	 Protocol  Timestamp
# Flow Duration	   Total Fwd Packets	 Total Backward Packets  	Total Length of Fwd Packets	 Total Length of Bwd Packets	 
# Fwd Packet Length Max	   Fwd Packet Length Min	 Fwd Packet Length Mean	 Fwd Packet Length Std
# Bwd Packet Length Max	 Bwd Packet Length Min	 Bwd Packet Length Mean	   Bwd Packet Length Std
# Flow Bytes/s	 Flow Packets/s	 Flow IAT Mean	 Flow IAT Std	 Flow IAT Max	Flow IAT Min	
# Fwd IAT Total	 Fwd IAT Mean	 Fwd IAT Std	 Fwd IAT Max	 Fwd IAT Min	
# Bwd IAT Total	 Bwd IAT Mean	 Bwd IAT Std	 Bwd IAT Max	 Bwd IAT Min	
# Fwd PSH Flags	 Bwd PSH Flags	 Fwd URG Flags	 Bwd URG Flags	 
# Fwd Header Length	  Bwd Header Length	    Fwd Packets/s	 Bwd Packets/s
# Min Packet Length	 Max Packet Length	 Packet Length Mean	    Packet Length Std	   Packet Length Variance
# FIN Flag Count	 SYN Flag Count	 RST Flag Count	 PSH Flag Count	 ACK Flag Count	
# URG Flag Count	 CWE Flag Count	 ECE Flag Count	 
# Down/Up Ratio	   Average Packet Size    Avg Fwd Segment Size	 Avg Bwd Segment Size	
# Fwd Header Length	  Fwd Avg Bytes/Bulk	 Fwd Avg Packets/Bulk	Fwd Avg Bulk Rate	 
# Bwd Avg Bytes/Bulk	 Bwd Avg Packets/Bulk	Bwd Avg Bulk Rate	
# Subflow Fwd Packets	 Subflow Fwd Bytes	 Subflow Bwd Packets	 Subflow Bwd Bytes	
# Init_Win_bytes_forward	 Init_Win_bytes_backward act_data_pkt_fwd	 min_seg_size_forward	
# Active Mean	 Active Std	 Active Max	 Active Min	Idle Mean	 Idle Std  Idle Max	 Idle Min	 
# Label

#--------------------------------------------------------------------------------
class cDataEncoding(metaclass=ABCMeta):
    """ Abstract class that defines parmaters require to parse raw data into desired encoding schemes
    """
    BOOL_DICT  = {'True': True, 'False': False} 

    INTERNAL_NETWORK_IPs = ['192.168.10.50', '192.168.10.51', '192.168.10.19','192.168.10.17','192.168.10.16','192.168.10.12','192.168.10.9','192.168.10.5',
                            '192.168.10.8',  '192.168.10.14', '192.168.10.15','192.168.10.25']

    PUBLIC_FACING_IPs = ['205.174.165.68', '205.174.165.66']

    def __init__(self):
        pass

    @property
    @abstractmethod
    def FILENAME(self):
        pass

    @property
    @abstractmethod
    def COL_TO_ONEHOT(self):
        pass

    @property
    @abstractmethod
    def COL_NO_FORMAT(self):
        pass

    @property
    @abstractmethod
    def NORM_METHOD(self):
        pass

    @property
    @abstractmethod
    def COL_TO_NORM(self):
        pass

    @property
    @abstractmethod
    def COL_FOR_SCORING(self):
        pass

    @property
    @abstractmethod
    def LABEL_MAPPING(self):
        pass

    @property
    @abstractmethod
    def COL_TO_USE(self):
        pass

#--------------------------------------------------------------------------------
class cAutoEncoding(cDataEncoding):
    FILENAME = "combined_test_data.pkl"

    COL_TO_ONEHOT = ["Protocol"]

    COL_NO_FORMAT = []

    NORM_METHOD = preprocessing.MinMaxScaler()

    COL_TO_NORM =   ["Destination Port", "Flow Duration", "Total Fwd Packets","Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets",
                     "Fwd Packet Length Mean",	"Fwd Packet Length Std", 
                     "Fwd IAT Mean", "Fwd IAT Std", "Flow IAT Max",	"Flow IAT Min", "Bwd IAT Mean","Bwd IAT Std", "Packet Length Mean","Packet Length Std",
                     "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", 
                     "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
                     "Average Packet Size", "Avg Fwd Segment Size", 
                     "Subflow Fwd Packets", "Subflow Fwd Bytes",  
                     "Init_Win_bytes_forward", "min_seg_size_forward",
                     "Active Mean", "Active Std", "Active Max", "Active Min"]

    COL_FOR_SCORING = ["Label", "Timestamp", "Full Label"]

    LABEL_MAPPING = {'BENIGN': False, 'Bot': True, 'DDoS': True, 'DoS GoldenEye': True, 'DoS Hulk': True, 'DoS Slowhttptest': True, 
                     'DoS slowloris': True, 'FTP-Patator': True, 'Heartbleed': True, 'Infiltration': True,
                     'PortScan': True, 'SSH-Patator': True, "Web Attack \x96 Brute Force": True, "Web Attack \x96 Sql Injection": True, "Web Attack \x96 XSS": True}

    COL_TO_USE = COL_TO_ONEHOT + COL_TO_NORM + COL_FOR_SCORING


#--------------------------------------------------------------------------------
class cPCAEncoding(cDataEncoding):
    FILENAME = "PCA_data.pkl"

    COL_TO_ONEHOT = ["Protocol"]

    COL_NO_FORMAT = []

    NORM_METHOD = preprocessing.MinMaxScaler()

    COL_TO_NORM =   ["Destination Port", "Flow Duration", "Total Fwd Packets","Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets",
                     "Fwd Packet Length Mean",	"Fwd Packet Length Std", 
                     "Fwd IAT Mean", "Fwd IAT Std", "Flow IAT Max",	"Flow IAT Min", "Bwd IAT Mean","Bwd IAT Std", "Packet Length Mean","Packet Length Std",
                     "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", 
                     "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
                     "Average Packet Size", "Avg Fwd Segment Size", 
                     "Subflow Fwd Packets", "Subflow Fwd Bytes",  
                     "Init_Win_bytes_forward", "min_seg_size_forward",
                     "Active Mean", "Active Std", "Active Max", "Active Min"]

    COL_FOR_SCORING = ["Label", "Timestamp", "Full Label"]

    LABEL_MAPPING = {'BENIGN': 'BENIGN', 'Bot': 'Bot', 'DDoS': 'DDoS', 'DoS GoldenEye': 'DoS GoldenEye', 'DoS Hulk': 'DoS Hulk', 
                     'DoS Slowhttptest': 'DoS Slowhttptest', 'DoS slowloris': 'DoS slowloris', 'FTP-Patator': 'FTP-Patator', 
                     'Heartbleed': 'Heartbleed', 'Infiltration': 'Infiltration', 'PortScan': 'PortScan', 'SSH-Patator': 'SSH-Patator', 
                     'Web Attack - Brute Force': 'Web Attack - Brute Force', 
                     'Web Attack - Sql Injection': 'Web Attack - Sql Injection', 'Web Attack - XSS': 'Web Attack - XSS'}

    COL_TO_USE = COL_TO_ONEHOT + COL_TO_NORM + COL_FOR_SCORING

#------------------------------------------------------------------------------
#Create instances of these classes to use
AUTOENCODER_PREPROCESS = cAutoEncoding()
PCA_PREPROCESS = cPCAEncoding()