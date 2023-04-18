import numpy as np
import pandas as pd

class FluxVector():
    """Class to store a flux vector associated with a ReactionNetwork.

    Args:
        model (bayflux.ReactionNetwork): The model this flux vector is associated with.
        fluxes (numpy.array, optional): An array of np.float64 fluxes in the same order as model.variables. Can also optionally be a dict of 2-tuples, where keys are reaction ids and values represent (forward flux, backwards flux) 

    Attributes:
        model (bayflux.ReactionNetwork): The model this flux vector is associated with.
        fluxes (numpy.array, optional): An array of np.float64 fluxes in the same order as model.variables. 

    """

    def __init__(self, model, fluxes=False):
        if type(fluxes) == dict:
            # if fluxes are passed as a dict, we will reformat them into
            # a 1D vector (numpy array) with the correct order
            
            # setup a mapping from reaction id to forward and reverse variables
            variableIndex = {v: index for index, v in enumerate(model.variables)}
            forwardIndex = {r.id: variableIndex[r.forward_variable] for r in model.reactions}
            reverseIndex = {r.id: variableIndex[r.reverse_variable] for r in model.reactions}
            
            fluxArray = np.zeros(len(model.variables))
            for reactionName, fluxPair in fluxes.items():
                assert reactionName in forwardIndex, 'Reaction name not in model'
                assert len(fluxPair) == 2, 'Each reaction must have a length 2 flux (reverse, forward)'
                assert fluxPair[0] >= 0.0, 'Invalid negative flux, reverse fluxes are a separate variable'
                assert fluxPair[1] >= 0.0, 'Invalid negative flux, reverse fluxes are a separate variable'
                fluxArray[forwardIndex[reactionName]] = fluxPair[0]
                fluxArray[reverseIndex[reactionName]] = fluxPair[1]
                
                if not model.reactions.get_by_id(reactionName).lower_bound < 0.0:
                    assert fluxPair[1] == 0.0, 'A non-reversible reaction had a non-zero reverse flux'
            fluxes = fluxArray
        
        assert len(fluxes) == len(model.variables), 'Number of variables in model does not match flux vector'
        self.fluxes = np.array(fluxes)
        self.model = model
        
    def to_Series(self):
        """convert the flux vector into a Pandas Series object.

        Returns:
            A pandas.Series object where each index represents a model variable name, and each value is a 64 bit flux for that variable.

        """
        # convert the flux vector into a Pandas Series object
        
        return pd.Series(self.fluxes, index=[v.name for v in self.model.variables])
    
    def __getitem__(self, key):
        return self.fluxes[key]
    
    def __repr__(self):
        return repr(self.to_Series())
    
    def _repr_html_(self):
        # to print HTML (e.g. for Jupyter) we convert to a Pandas DataFrame
        # so we can hijack it's HTML output features
        return pd.DataFrame(self.to_Series(), columns=['Flux'])._repr_html_()
