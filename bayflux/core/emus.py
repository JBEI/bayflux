import numpy as np
import scipy
import itertools
from copy import copy
from emusFortran import executeemuoperationsfortran
from bayflux.core.fluxes import FluxVector
# from pypardiso import PyPardisoSolver

class EMU():
    """A class for representing EMUs that cross-references a metabolite object.

    Args:
        metabolite (cobra.Metabolite): A COBRApy metabolite object, which
            this EMU is a subset of.
        indices (list): A list of integers representing the 1-indexed
            atoms in the metabolite to include in this EMU. Our standard
            is to use canonical smiles, and represent atoms from left to right.

    Attributes:
        metabolite (cobra.Metabolite): A COBRApy metabolite object, which
            this EMU is a subset of.
        indices (list): A list of integers representing the 1-indexed
            atoms in the metabolite to include in this EMU. Our standard
            is to use canonical smiles, and represent atoms from left to right.

    """

    def __init__(self, metabolite, indices):
        self.metabolite = metabolite # a COBRApy metabolite object
        self.indices = sorted(indices) # atom numbers, starting at one

    def __repr__(self):
        return str(self.metabolite) + str(self.indices)

    def size(self):
        return len(self.indices)

    def __hash__(self):
        # produce a unique hash key for each emu, to check if EMUs are duplicated or not
        # here we add the metabolite id into the hash, because at time of
        # writing cobrapy has no hash function for metabolite, so the memory
        # address is used which gives very little diversity of hash values
        uniqueTuple = (self.metabolite, self.metabolite.id,) + tuple(self.indices)
        return hash(uniqueTuple)

    def __eq__(self, other):
        if isinstance(other, EMU):
            return hash(self) == hash(other)
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))

class EMUTransition():
    """A class for representing an EMU reaction.

    Args:
        reactants (list): A list of EMU objects representing all reactants.
        product (EMU): A single EMU object representing the product.
        parentReaction (cobra.Reaction): The parent reaction this EMU reaction
            is derived from. Used for applying the correct flux later on.
        forward (bool): True if this reaction comes from the forward direction
            of the parent reaction.
        contribution (float): The fraction of the parent reaction flux carried
            by this EMU reaction within a set of equivalent EMU reactions.

    Attributes:
        reactants (list): A list of EMU objects representing all reactants.
        product (EMU): A single EMU object representing the product.
        parentReaction (cobra.Reaction): The parent reaction this EMU reaction
            is derived from. Used for applying the correct flux later on.
        forward (bool): True if this reaction comes from the forward direction
            of the parent reaction.
        contribution (float): The fraction of the parent reaction flux carried
            by this EMU reaction within a set of equivalent EMU reactions.
    """

    def __init__(self, reactants, product, parentReaction, forward=True, contribution=1.0):

        assert len(reactants) > 0, 'no reactants'

        self.reactants = reactants
        self.product = product
        self.contribution = contribution
        self.parentReaction = parentReaction
        self.forward = forward

    def __repr__(self):
        if self.forward:
            contribution = str(self.contribution)
        else:
            contribution = '-' + str(self.contribution)
        return contribution + "*" + self.parentReaction.id + ': ' + ', '.join([repr(r) for r in self.reactants]) +\
            '-> ' + repr(self.product)

    def copy(self):
        """Copies EMU transition without copying the parent reaction
            or metabolites.
        """
        return EMUTransition(
                reactants=copy(self.reactants), 
                product=copy(self.product), 
                parentReaction=self.parentReaction,  
                forward=copy(self.forward), 
                contribution=copy(self.contribution), 
            )

def findProducingReactions(metabolite, ignoreNoTransitions=True):
    """Find all reactions which can produce a given metabolite in the forward
        and reverse directions.

    Args:
        metabolite (cobra.Metabolite): A metabolite to look up.
        ignoreNoTransitions (bool, optional): If True, only EnhancedReaction
            objects that have 1 or more atom transitions will be returned.

    Returns:
        list: A list of tuples (bayflux.EnhancedReaction, bool) representing each
            reaction that can produce the input metabolite. The bool value
            is true if and only if the metabolite is produced in the
            forward direction. If a metabolite can be produced in both directions
            it will occur twice, once with a True, and once with a False value.
    """
    
    allReactions = list(metabolite.reactions)
    producingReactions = []
    reverseProducingReactions = []
    
    for reaction in allReactions:
        if ignoreNoTransitions and (len(reaction.transitions) == 0):
            # skip this reaction if it has no transitions
            continue

        if metabolite in reaction.products:
            producingReactions.append((reaction, True))
        
        if (reaction.lower_bound < 0.0) and (metabolite in reaction.reactants):
            producingReactions.append((reaction, False))
            
    return producingReactions

def findProducingEMUTransitions(inputEMU):
    """Find all EMUTransition objects that directly produce the inputEMU.

    Args:
        inputEMU (EMU): An EMU to look up.

    Returns:
        list: A list of EMUtransition objects.
    """
    
    # process input EMU
    metabolite = inputEMU.metabolite
    
    EMUtransitions = []
    
    # get all reactions producing this metabolite
    producingReactions = findProducingReactions(metabolite)
    
    for reaction, direction in producingReactions:
        transitionList = reaction.transitions
        contribution = 1.0 / len(transitionList)
        
        # if the reaction is reversed, reverse the transition as well
        if not direction:
            transitionList = [t.reverse() for t in transitionList]
        
        for transition in transitionList:
            # get atom indices for our metabolite in this transition
            allIndicesInMetabolite = [i[1] for i in list(filter(lambda x: x[0] == metabolite, transition.products))]
            
            # get subset of indices which match our inputEMU
            inputEMUIndices = [[x[i - 1] for i in inputEMU.indices] for x in allIndicesInMetabolite]
            
            # loop over each occurrence of the metabolite and find which EMU reactions create those
            for indices in inputEMUIndices:
                reactantEMUs = []
                for reactant in transition.reactants:
                    
                    # get overlap between this reactant and EMU
                    intersect = set(indices).intersection(reactant[1])
                    
                    if len(intersect) > 0:
                        # convert intersect labels into positions
                        positions = []
                        for atom in list(intersect):
                            positions.append(reactant[1].index(atom) + 1)
                        reactantEMUs.append(EMU(reactant[0], positions))
                        
                # create EMU transition object
                EMUtransitions.append(
                    EMUTransition(reactants=reactantEMUs, 
                                  product=inputEMU, 
                                  parentReaction=reaction, 
                                  forward=direction, 
                                  contribution=contribution)
                )
    return EMUtransitions

def emuDecomposition(inputEMUs, transitionLimit=5000000):
    """Perform a full EMU decomposition of the input EMUs.

    Args:
        inputEMUs (list): A list of target EMUs to analyze.
        transitionLimit (int): The maximum number of atom transitions to allow
            in the deconstruction. This should be increased for very large
            models on high memory systems.

    Returns:
        list: A list of EMUTransition objects representing the full decomposition.
    """
    
    unvisitedEMUs = copy(inputEMUs) # the stack of EMUs to investigate
    visitedEMUs = set(inputEMUs) # previously investigated EMUs 
    transitionList = [] # these are the emu transitions we are going to return
    
    while len(unvisitedEMUs) > 0: # visit all unvisited EMUs
        thisEMU = unvisitedEMUs.pop()
        newRxns = findProducingEMUTransitions(thisEMU)
        if len(newRxns) > 0:
            transitionList.extend(newRxns)
            
            newEMUs = [r.reactants for r in newRxns] # get all reactants for these EMU reactions
            
            for EMU in itertools.chain.from_iterable(newEMUs): # visit any new reactant EMUs later
                if EMU not in visitedEMUs:
                    unvisitedEMUs.append(EMU)
                    visitedEMUs.add(EMU) # mark this EMU as visited

            # check that we're not stuck in an endless loop
            assert len(transitionList) <= transitionLimit, 'too many transitions found, increase transitionLimit'
    return transitionList

def EMUsizes(EMUTransitionList):
    """Identify which size EMUs are found in a list of EMUtransitions.

    Args:
        EMUTransitionList (list): A list of EMUtransition objects.

    Returns:
        list: A list of integers in sorted increasing order, which represents
            all EMU sizes found in the products and reactants of the input list.
    """

    sizes = set()
    for EMUTransition in EMUTransitionList:
        sizes.update([emu.size() for emu in EMUTransition.reactants])
        sizes.update([EMUTransition.product.size()])
    return sorted(sizes)

def splitBySize(EMUTransitionList):
    """Split apart an EMU transition list by size.

    Args:
        EMUTransitionList (list): A list of EMUtransition objects.

    Returns:
        dict: A dict where keys represent EMU transition product sizes in the
            input list, and values are a list of EMUTransition objects
            from the input which produce a product of this size.
    """

    allSizes = EMUsizes(EMUTransitionList)
    splitBySize = dict.fromkeys(allSizes, [])
    for t in EMUTransitionList:
        size = t.product.size()
        splitBySize[size] = splitBySize[size] + [t]
    return(splitBySize)

def constructMatrixCoords(EMUTransitionList):
    """Computes human readable and hash versions of the list of internal EMUs
        and external EMUs for a list of EMUTransitions.

    Args:
        EMUTransitionList (list): A list of EMUtransition objects, typically
            of equal product size.

    Returns:
        dict: A dict of dicts representing the internal and external EMUs
            with the keys 'internalEMUs' and 'externalEMUs' respectively.
            Each of these values is a dict which has three keys 'text', 'hashes',
            and 'objects' with list values (in the same order for all 3)
            representing human readable text, a hash value, and the EMU objects
            themselves respectively. For condensation EMUs, the 'externalEMUs'
            objects will be list form, where each element is an EMU object.
    """

    hashToEMUrepr = {}
    hashToEMUobjects = {}
    
    reactants = set()
    products = set()
    
    for transition in EMUTransitionList:
        if len(transition.reactants) > 1:
            # if this is a convolution transition, create text with multiplier
            reactantHash = hash(tuple(transition.reactants))
            reactantListText = ' x '.join([repr(r) for r in transition.reactants])
            hashToEMUrepr[reactantHash] = reactantListText
            hashToEMUobjects[reactantHash] = transition.reactants
            reactants.add(reactantHash)
        else:
            reactantHash = hash(transition.reactants[0])
            hashToEMUrepr[reactantHash] = repr(transition.reactants[0])
            hashToEMUobjects[reactantHash] = transition.reactants[0]
        
        reactants.add(reactantHash)
        
        productHash = hash(transition.product)
        hashToEMUrepr[productHash] = repr(transition.product)
        hashToEMUobjects[productHash] = transition.product
        products.add(productHash)
    
    # create internal hashes, text, and object lists in the same order
    internalEMUHashes = list(products)
    # sort like Antoniewicz et al.
    internalEMUHashes = sorted(internalEMUHashes, key=lambda h: (hashToEMUobjects[h].indices[0], hashToEMUobjects[h].metabolite.id))
    internalEMUrepr = [hashToEMUrepr[h] for h in internalEMUHashes]
    internalEMUobjects = [hashToEMUobjects[h] for h in internalEMUHashes]
    
    # for reordering, create a special hash of convolution reactions that only has the first EMU
    firstElementOnly = {}
    for k, v in hashToEMUobjects.items():
        if type(v) == list:
            firstElementOnly[k] = v[0]
        else:
            firstElementOnly[k] = v
    
    # create external hashes, text, and object lists in the same order
    reactants.difference_update(products)
    externalEMUHashes = list(reactants)
    # sort like Antoniewicz et al.
    externalEMUHashes = sorted(externalEMUHashes, key=lambda h: (firstElementOnly[h].indices[0], firstElementOnly[h].metabolite.id))
    externalEMUrepr = [hashToEMUrepr[h] for h in externalEMUHashes]
    externalEMUobjects = [hashToEMUobjects[h] for h in externalEMUHashes]
    
    return {
        'internalEMUs': {'text': internalEMUrepr, 'hashes': internalEMUHashes, 'objects': internalEMUobjects},
        'externalEMUs': {'text': externalEMUrepr, 'hashes': externalEMUHashes, 'objects': externalEMUobjects},
    }

def constructZeroMatrices(matrixCoords):
    """Constructs the A and B zero matrices from the results of constructMatrixCoords.

    Args:
        matrixCoords (dict): the result of running constructMatrixCoords()

    Returns:
        tuple: A tuple of length 2, where each element is a numpy.ndarray representing
            an all zeros version of the appropriately sized A or B matrices
            for EMU simulation for later 'injection' of fluxes by addition.
    """
    
    internalEMUs = matrixCoords['internalEMUs']['text']
    externalEMUs = matrixCoords['externalEMUs']['text']
    
    matA = np.zeros(shape=(len(internalEMUs), len(internalEMUs)), order='F')
    matB = np.zeros(shape=(len(internalEMUs), len(externalEMUs)), order='F')
    
    return (matA, matB)

def mergeDuplicates(transitionList):
    # if two EMU transitions have identical products and reactants, but just different
    # fluxes, here we will merge them into one
    
    transitionDict = {}
    
    for transition in transitionList:
        reactantHash = tuple(sorted([hash(emu) for emu in transition.reactants]))
        totalHash = hash((reactantHash, transition.product, transition.parentReaction, transition.forward,))
        
        # add each transition to the dict where keys represent the hashes of everything but contribution
        if totalHash in transitionDict:
            transitionDict[totalHash].append(transition)
        else:
            transitionDict[totalHash] = [transition]
    
    # create a new contributions list, where transitions that differ only the the contribution
    # are merged into one, with a new contribution equal to the sum of previous ones
    newTransitions = []
    for hashKey, transitionList in transitionDict.items():
        if len(transitionList) == 1:
            newTransitions.append(transitionList[0])
        else:
            contributionSum = 0
            for transition in transitionList:
                contributionSum += transition.contribution
            mergedTransition = transitionList[0].copy()
            mergedTransition.contribution = contributionSum
            newTransitions.append(mergedTransition)
    
    return newTransitions

def _traceReplacement(emu, replacementPairs):
    # recursively replace all values in a replacementDict with the final end product

    visited = set()
    
    while emu in replacementPairs:
        emu = replacementPairs[emu]

        # if we get back to an EMU we already visited, it means there is a circular reference
        # which should never happen
        assert emu not in visited, (
                                   'circular loop with no inputs ' 
                                   'found in transitions, cannot prune network. ' 
                                   'EMUs involved: ' + ' '.join([repr(x) for x in visited]))

        visited.add(emu)
    return emu

def _identifyIdenticalEMUs(transitionList, metabolitesToKeep):
    # returns a dict of EMUs (keys) which can be deleted from the EMU network with no information loss
    # where each key is the EMU they should be replaced with
    # input EMUs are ignored so they do not get replaced
    
    # create a dict where keys are product EMUs, and values are all possible EMUs which
    # create it. This lets us then check if there's only one EMU which creates it, so they
    # can be merged, as they will have identical labeling.
    producingReactants = {}
    
    for transition in transitionList:
        if transition.product in producingReactants:
            producingReactants[transition.product].update(transition.reactants)
        else:
            producingReactants[transition.product] = set(transition.reactants)
            
    # for each product which only has one producing EMU, create
    # a dict where keys represent the reactant EMU that should be replaced with values (product EMUs)
    replacementPairs = {value.pop(): key for (key, value) in 
        filter(lambda x: len(x[1]) == 1, producingReactants.items())}

    # remove any input and output EMUs that ended up in this dict as keys
    replacementPairs = {key: value for (key, value) in
        filter(lambda x: x[0].metabolite not in metabolitesToKeep, 
        replacementPairs.items())}
    
    # trace through the graph to create a dict where keys should be replaced with
    # values in all instances to prune the network
    finalReplacementPairs = {}
    for key, value in replacementPairs.items():
        finalReplacementPairs[key] = _traceReplacement(value, replacementPairs)
    
    return finalReplacementPairs

def pruneTransitionList(transitionList, metabolitesToKeep):
    # Using a dict of key, value pairs, replace all key emus with the value
    # emu in transitionList, and then return the updated transitionList

    replacementDict = _identifyIdenticalEMUs(transitionList, metabolitesToKeep)
    newTransitionsList = []

    for transition in transitionList:
        transition = transition.copy()

        # update products to new metabolite
        if transition.product in replacementDict:
            transition.product = replacementDict[transition.product]
            
        # update reactants to new metabolite
        for i in range(0, len(transition.reactants)):
            if transition.reactants[i] in replacementDict:
                transition.reactants[i] = replacementDict[transition.reactants[i]]

        # don't keep transitions that now only produce themselves
        if (len(transition.reactants) == 1) and (hash(transition.reactants[0]) == hash(transition.product)):
            continue
                    
        newTransitionsList.append(transition)

    return newTransitionsList

def compileEMUTransitionList(EMUTransitionList, matrixCoords, model):
    """Compiles an EMU transition list into a set of matrix addition operations indexed by position.

    Args:
        EMUTransitionList (list): A list of EMUtransition objects, typically
            of equal product size.
        matrixCoords (dict): the result of running constructMatrixCoords()
            on the above list.
        model (bayflux.ReactionNetwork): A genome scale model, used to determine the order of variables in a flux vector.

    Returns:
        list: A list of tuples, where each inner tuple represents a matrix 
        addition operation to the A or B matrix and has the following structure:
            -matrix selector (0: matrix A, 1: matrix B)
            -row: row number to inject to into in the above matrix
            -column: column number to inject into in the above matrix
            -reaction selector: a column number in the flux vector
            -multiplier: a float: multiply flux by this before injecting
    """
    
    # here we create dict lookup tables of position vectors to make this FAST
    internalEMUPositions = {emu: position for position, emu in enumerate(matrixCoords['internalEMUs']['hashes'])}
    externalEMUPositions = {emu: position for position, emu in enumerate(matrixCoords['externalEMUs']['hashes'])}
    
    # setup a mapping from reaction id to forward and reverse variables
    variableIndex = {v: index for index, v in enumerate(model.variables)}
    forwardIndex = {r: variableIndex[r.forward_variable] for r in model.reactions}
    reverseIndex = {r: variableIndex[r.reverse_variable] for r in model.reactions}

    # now perform decomposition
    operations = []
    for thisEMUTransition in EMUTransitionList:
        
        # determine the flux vector position
        if thisEMUTransition.forward:
            fluxVectorPos = forwardIndex[thisEMUTransition.parentReaction]
        else:
            fluxVectorPos = reverseIndex[thisEMUTransition.parentReaction]
        
        # hash the reactant EMU(s)
        if len(thisEMUTransition.reactants) == 1:
            reactantHash = hash(thisEMUTransition.reactants[0])
        else:
            reactantHash = hash(tuple(thisEMUTransition.reactants))
            
        # hash the product EMU
        productHash = hash(thisEMUTransition.product)
        
        # determine if this should go in the A or B matrix
        if reactantHash in externalEMUPositions: # B matrix
            operations.append(( # subtract flux from input (reactant) in B matrix
                1, # B matrix
                internalEMUPositions[productHash], # row index
                externalEMUPositions[reactantHash], # column index
                fluxVectorPos, # reaction selector
                -thisEMUTransition.contribution, # flux contribution
            ))
        elif reactantHash in internalEMUPositions and productHash in internalEMUPositions: # A matrix
            operations.append((
                0, # A matrix
                internalEMUPositions[productHash],
                internalEMUPositions[reactantHash],
                fluxVectorPos, # reaction selector
                thisEMUTransition.contribution, # flux contribution
            ))    
        else: 
            raise Exception('reactant or product hash not in A or B matrix')
            
        operations.append((
            0, # A matrix
            internalEMUPositions[productHash],
            internalEMUPositions[productHash],
            fluxVectorPos, # reaction selector
            -thisEMUTransition.contribution, # flux contribution
        ))
    return np.array(operations, order='F')

def executeCompiledEMUTransitionList(operations, zeroMatrices, fluxes):
    """Execute the matrix addition operations specified in the compiled EMU decomposition
        and return the new A and B matrices.

    Args:
        operations (list): A list of compiled operations as produced by
            compileEMUTransitionList.
        zeroMatrices (tuple): A tuple of length 2, where each element is a numpy.ndarray representing
            an all zeros version of the appropriately sized A or B matrices
            for EMU simulation for later 'injection' of fluxes by addition.
        fluxes (bayflux.FluxVector): A flux vector or numpy array with a flux value for each variable in the genome scale model.
            The order of values in fluxVector should match
            that used to generate the operations list.

    Returns:
        tuple: A tuple of length 2, where each element is a numpy.ndarray representing
            the A and B matrices respectively, after execution of the
            specified addition operations.
    """

    # copy the input zeroMatrices since we will modify them in place
    outputMatrices = [np.copy(m) for m in zeroMatrices]
    
    # if we got a FluxVector as input, keep just the numpy array inside
    if type(fluxes) == FluxVector:
        fluxes = fluxes.fluxes
    
    # generate A and B matrices
    # Note: This is a Fortran function which modifies matA and matB in place
    executeemuoperationsfortran(outputMatrices[0], outputMatrices[1], operations, fluxes)

    return outputMatrices

def executeCompiledMatrixY(substrateMDVs, operationsY, results, size):
    matY = np.zeros(shape=(len(operationsY), size + 1), order='F')
    
    for rowCounter, row in enumerate(operationsY): # loop over operations
        for i, operation in enumerate(row):
            if operation[0] == 0: # get input EMU MDV
                labeling = substrateMDVs[operation[1]]
            else: # get previously calculated MDV
                # labeling = results[operation[0]][operation[1]].tolist()[0]
                labeling = results[operation[0]][operation[1]]
                
            if i == 0:
                allLabeling = labeling
            else: # if this isn't the first, convolve it with the others
                allLabeling = np.convolve(allLabeling, labeling)
        matY[rowCounter] = allLabeling
    
    return matY

def extractSubstrateEMU(inputEMU, substrateLabelingDict):
    """Compute the MDV for a given EMU from a set of labeled substrates.

    Args:
        inputEMU (EMU): An EMU object representing a subset of one of the labeled 
            substrate metabolites.
        substrateLabelingDict (dict): A dict where keys are cobra.Metabolite
            objects, and values a tuple of length 2 tuples each representing
            a fractional labeling pattern on an input substrate. The first value
            in the length 2 tuple is a float representing a fractional contribution
            and the second is a list of integers representing 
            the number of extra neutrons at each atom location. The standard index
            is left to right in the canonical SMILES structure of the metabolite.
            Fractional contributions must sum to 1.0.

    Returns:
        list: A list of float values, representing the mass distribution vector
            for the inputEMU, starting from the fraction with 0 extra neutrons
            in the first position.
    """
    
    # create zero length vector for MID of this EMU
    emuLabeling = [0] * (len(inputEMU.indices) + 1)
    contributionSum = 0.0

    for contribution, labelingVector in substrateLabelingDict[inputEMU.metabolite]:
        # extract the labeling for the subset of this molecule in the EMU
        labelingInThisEMU = [labelingVector[i - 1] for i in inputEMU.indices]

        contributionSum += contribution
        
        # inject the fractional contribution at the sum of labeled atoms
        emuLabeling[sum(labelingInThisEMU)] += contribution

    assert contributionSum == 1.0, 'sum of labeling contributions not 1.0 for metabolite ' + str(inputEMU.metabolite)
    return np.array(emuLabeling)

def compileMatrixY(compiledData, matrixCoords, size, substrateLabelingDict):
    """
    Returns: 1) a set of input MDVs in tuple of tuples form
    
    2) A set of compiled operations with the following syntax:
    
    A list where each element corresponds to a row in matY in order.
    
    Each of these is a length 2 list with:
    -result size, representing a matrix X EMU of a particular size, or an input emu (called size 0)
    -MDV number: a row number from that size EMU result, or from the input MDVs in tuple form
    
    Execution instructions:
    convolve together all MDVs with the same row #, and inject the result into that row
    """
    substrateMDVs = []
    allOperations = []
    
    externalEMUs = matrixCoords['externalEMUs']
    
    rowNumber = 0
    for externalEMUs in externalEMUs['objects']: # loop over the external EMUs
        if type(externalEMUs) is not list: # if this is a single EMU place it in a list
            externalEMUs = [externalEMUs]
        operations = []
        
        for emu in externalEMUs: # loop over all EMUs to convolve together
            
            if emu.metabolite in substrateLabelingDict: # if this is an input substrate
                massDistribution = extractSubstrateEMU(emu, substrateLabelingDict)
                operations.append([
                    0, # input/substrateEMUs EMUs
                    len(substrateMDVs) # the position in substrateEMUs
                    ])
                substrateMDVs.append(massDistribution)
            else: # not an input substrate
                emuSizeIndex = [x['size'] for x in compiledData].index(emu.size())
                resultIndex = compiledData[emuSizeIndex]['matrixCoords']['internalEMUs']['hashes'].index(hash(emu))
                operations.append([
                    emu.size(), # which previous size EMU is this?
                    resultIndex # the position in substrateEMUs
                    ])            
            
        allOperations.append(operations)
    
    return substrateMDVs, allOperations
    
def emuCompile(emusToSimulate, model, substrateLabelingDict, merge=False, prune=False):
    """Pre-deconstructs an EMU network, for later simulation of specific
        flux vectors.

    Args:
        emusToSimulate (list): A list of EMU objects, representing the
            metabolites we would like to predict MDVs for.
        reactions (list): A list of 'str' objects representing the names
            of reactions in our flux vector in order.

    Returns:
        list: A list of dicts representing each EMU size sub-network with the
            following keys and values:
                'matrixCoords': (dict) the output of constructMatrixCoords
                'size': (int) the EMU sub-network size
                'operations': (list) the output of compileEMUTransitionList 
    """

    compiled = []
    
    # decompose network into EMU transitions 
    EMUTransitionList = emuDecomposition(emusToSimulate)

    if merge:
        # merge transitions with the same products, reactants, and
        # reactions by adding contributions
        EMUTransitionList = mergeDuplicates(EMUTransitionList)

    if prune:
        metabolitesToKeep = set()

        # add output metabolites so they don't get pruned
        for emu in emusToSimulate:
            metabolitesToKeep.add(emu.metabolite)

        # add labeling metabolites so they also don't get pruned
        metabolitesToKeep.update(substrateLabelingDict.keys())

        # prune EMU nodes that just pass through and have only
        # one input
        EMUTransitionList = pruneTransitionList(EMUTransitionList, metabolitesToKeep)
    
    # get sorted list of all EMU sizes in the decomposition
    allSizes = EMUsizes(EMUTransitionList)

    # split the EMU transitions apart by product size
    transitionsBySize = splitBySize(EMUTransitionList)

    for size in allSizes:
        emuTransitionSubset = transitionsBySize[size]
        
        # construct matrix coordinates for A and B matrices
        matrixCoords = constructMatrixCoords(emuTransitionSubset)

        # skip this size if it has no internal EMUs
        if len(matrixCoords['internalEMUs']['objects']) == 0:
            continue

        # compile transitions list into a set of matrix addition operations
        operations = compileEMUTransitionList(emuTransitionSubset, matrixCoords, model)

        # compile matrix Y operations
        substrateMDVs, operationsY = compileMatrixY(compiled, matrixCoords, size, substrateLabelingDict)

        compiled.append({
            'size': size, 
            'matrixCoords': matrixCoords, 
            'operations': operations, 
            'substrateMDVs': substrateMDVs,
            'operationsY': operationsY,
        })
    
    return compiled

def simulateLabeling(compiledData, fluxes, substrateLabelingDict):
    """Simulates the Mass Distribution Vector (MDV) for a set of inputEMUs given
        a specific flux vector.

    Args:
        compiledData (list): The output of emuCompile.
        fluxes (numpy.matrix): A matrix of fluxes where columns represent
            reactions, the first row is forward fluxes, and the second row is
            backwards fluxes. Non-reversible reactions should always have
            a flux of 0 in the second row. The order of columns should match
            that used to generate the operations list.
        substrateLabelingDict (dict): See bayflux.extractSubstrateEMU for 
            a detailed description.

    Returns:
        dict: A dict of results where each key is an integer specifying
            an EMU sub-network size, and each value is an EMU simulation X
            numpy.matrix which contains simulated MDVs.
            Columns represent number of extra 
            neutrons starting at 0, while rows represent the 'internalEMUs' from
            'matrixCoords' in the compiledData.
    """
    
    results = {}
    
    for subset in compiledData:
        size = subset['size']
        matrixCoords = subset['matrixCoords'] 
        operations = subset['operations']
        substrateMDVs = subset['substrateMDVs']
        operationsY = subset['operationsY']
        matA, matB = constructZeroMatrices(matrixCoords)

        # if we got a FluxVector as input, keep just the numpy array inside
        if type(fluxes) == FluxVector:
            fluxes = fluxes.fluxes
        
        # generate A and B matrices
        # Note: This is a Fortran function which modifies matA and matB in place
        executeemuoperationsfortran(matA, matB, operations, fluxes)
        
        # generate Y matrix on the fly
        matY = executeCompiledMatrixY(substrateMDVs, operationsY, results, size)
        
        # compute predicted labeling
        matBY = matB.dot(matY)
        # matX = np.linalg.inv(matA).dot(matBY)

        # This method is much faster for large models, and slightly
        # slower for small models than linalg.inv above
        matX = np.linalg.solve(matA, matBY)

        # Solve w/ MKL pardiso
        # currently this is even slower...
        # sol = PyPardisoSolver()
        # sol.set_num_threads(2)
        # matAsparse = csc_matrix(matA)
        # matX = sol.solve(matAsparse, matBY)

        results[size] = np.array(matX, order='F')
    
    return results
        
