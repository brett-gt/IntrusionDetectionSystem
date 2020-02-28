""" 
"""

from abc import ABCMeta, abstractmethod

#--------------------------------------------------------------------------------
class cHyperParameters(metaclass=ABCMeta):
    """ Abstract class that defines parmaters require to create and train a network

          layers -> array defining the number of nodes for each layer.  Since the
        autoencoder has symetrical encoder and decoder sides, only the encoder
        layer is defined in the array (function automatically creates decoder).
        The final layer size is the bottleneck which is not mirrored.

        kernel_reg -> array defining the kernel regularization parameters for
        each layer.  

        drop_out -> array defining dropout parameters for each layer.  *Should have one
        more element than the number of layers because first entry is applied to input.
    """

    #TODO: Could put some input validation in (check sizes of arrays against each other)

    def __init__(self, model_name, layers, kernel_reg, drop_out):
        pass

    @property
    @abstractmethod
    def model_name(self):
        pass

    @property
    @abstractmethod
    def random_seed(self):
        pass

    @property
    @abstractmethod
    def num_epochs(self):
        pass

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @property
    @abstractmethod
    def early_term_period(self):
        pass

    @property
    @abstractmethod
    def layers(self):
        pass

    @property
    @abstractmethod
    def kernel_reg(self):
        pass

    @property
    @abstractmethod
    def drop_out(self):
        pass


#--------------------------------------------------------------------------------
class cAutoHyper(cHyperParameters):
    random_seed = 1234

    num_epochs = 200

    batch_size = 256

    early_term_period = 5

    def __init__(self, model_name, layers, kernel_reg, drop_out):
        self.model_name = model_name
        self.layers = layers
        self.kernel_reg = kernel_reg
        self.drop_out = drop_out
        pass

    def drop_out(self, value):
        self.drop_out = value

    def kernel_reg(self, value):
        self.kernel_reg = value

    def layers(self, value):
        self.layers = value

    def model_name(self, value):
        self.model_name = value




    

