import bayflux 
import pandas as pd
import numpy as np
import cobra
from collections import defaultdict

class MassDistribution():
    
    def __init__(self, model, data=None):
        if data == None:
            self.data = defaultdict(list)
        else:
            self.data = data
        self.model = model
        
    def __repr__(self):
        return repr(self.toDF())
        
    def toDF(self):
        # unroll multiple MDVs for the same EMU into separate rows and export
        df = pd.DataFrame(
            [[row[0].metabolite.id, ','.join([str(i) for i in row[0].indices])] + mdv 
                for row in self.data.items() 
                for mdv in row[1]])
        df.columns = ['metabolite', 'atoms'] + list(pd.RangeIndex(start=0, stop=df.shape[1] - 2, step=1))
        return df
    
    def writeToFile(self, outputFileName):
        self.toDF().to_csv(outputFileName, sep='\t', index=False)
        
def readMassDistribution(model, inputFileName, format='csv'):
    
    assert format in {'csv'}, 'unsupported format'
    
    # read the file
    massDistributionDF = pd.read_csv(
        inputFileName, 
        header=0, 
        index_col=0, 
        float_precision='round_trip',
        skip_blank_lines=True,
        comment='#',
        # names=[0,1,2],
        sep='\t')
    
    # create a dictionary for storing the MDV data
    dataDict = defaultdict(list)
    
    # iterate over rows and create MDV entries
    for row in massDistributionDF.iterrows():
        # get series for this row and unpack variables
        rowSeries = row[1]
        metaboliteId = rowSeries.name
        
        # get the metabolite
        metabolite = model.metabolites.get_by_id(metaboliteId)

        # get atom indices
        indices = [int(i) for i in rowSeries[0].split(',')]
        
        # get the mass distribution and confirm it matches the carbon count
        mdvList = list(rowSeries.dropna())[1:]
        assert len(mdvList) == len(indices) + 1, 'mismatch between mass inputs and carbon count in model for ' + str(metaboliteId)
        
        # create dict entry with a full sized EMU as the key
        dataDict[bayflux.EMU(metabolite, indices)].append(mdvList)
    
    return MassDistribution(model, dataDict)


