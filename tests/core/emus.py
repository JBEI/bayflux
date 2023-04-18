import cobra
import bayflux
import pandas as pd
import numpy as np
import pytest
from copy import copy, deepcopy 

# First we will use COBRApy to build a model corresponding to the reaction stoichiometry in Fig. 3 of Antoniewicz 2006 which consists of 6 metabolites named A-F, and 5 reactions

# create a blank model
model = cobra.Model('fig3')

# define metabolites
A = cobra.Metabolite(
    'A',
    formula='C3',
    name='A',
    compartment='c')
B = cobra.Metabolite(
    'B',
    formula='C3',
    name='B',
    compartment='c')
C = cobra.Metabolite(
    'C',
    formula='C2',
    name='C',
    compartment='c')
D = cobra.Metabolite(
    'D',
    formula='C3',
    name='D',
    compartment='c')
E = cobra.Metabolite(
    'E',
    formula='C1',
    name='E',
    compartment='c')
F = cobra.Metabolite(
    'F',
    formula='C3',
    name='F',
    compartment='c')

# define reactions and add them to the model
a_b = cobra.Reaction('a_b')
a_b.lower_bound = 0
a_b.upper_bound = 500
a_b.add_metabolites({
    A: -1.0,
    B: 1.0
})

b_ec = cobra.Reaction('b_ec')
b_ec.lower_bound = 0
b_ec.upper_bound = 500
b_ec.add_metabolites({
    B: -1.0,
    E: 1.0,
    C: 1.0
})

bc_de = cobra.Reaction('bc_de')
bc_de.lower_bound = 0
bc_de.upper_bound = 500
bc_de.add_metabolites({
    B: -1.0,
    C: -1.0,
    D: 1.0,
    E: 2.0
})

d_f = cobra.Reaction('d_f')
d_f.lower_bound = 0
d_f.upper_bound = 500
d_f.add_metabolites({
    D: -1.0,
    F: 1.0
})

b_d = cobra.Reaction('b_d')
b_d.lower_bound = -500
b_d.upper_bound = 500
b_d.add_metabolites({
    B: -1.0,
    D: 1.0
})

model.add_reactions([a_b, b_ec, bc_de, d_f, b_d])

# Now we convert this model to a bayflux ReactionNetwork
model = bayflux.ReactionNetwork(model)

# Here we set the atom transitions for each reaction, and show an example of viewing an EnhancedReaction which shows atom transitions.

# create dict of metabolites by name
# we use this instead of directly using the metabolite IDs above, because the conversion to
# a bayflux.ReactionNetwork created new metabolite objects
m = {m.id:m for m in model.metabolites}

model.reactions.a_b.transitions = [bayflux.AtomTransition(
        ((m['A'], [1,2,3]),), # reactant labels
        ((m['B'], [1,2,3]),) # product labels
    )]
model.reactions.b_ec.transitions = [bayflux.AtomTransition(
        ((m['B'], [1,2,3]),), # reactant labels
        ((m['E'], [1]), (m['C'], [2,3]),) # product labels
     )]
model.reactions.bc_de.transitions = [bayflux.AtomTransition(
        ((m['B'], [1,2,3]), (m['C'], [4,5]),), # reactant labels
        ((m['E'], [1]), (m['D'], [2,3,4]), (m['E'], [5]),) # product labels
     )]
model.reactions.d_f.transitions = [bayflux.AtomTransition(
        ((m['D'], [1,2,3]),), # reactant labels
        ((m['F'], [1,2,3]),) # product labels
     )]
model.reactions.b_d.transitions = [bayflux.AtomTransition(
        ((m['B'], [1,2,3]),), # reactant labels
        ((m['D'], [1,2,3]),) # product labels
     )]

def test_AtomTransition():
    assert repr(model.reactions.b_d.transitions[0]) == 'B --> D\tabc : abc', 'atom transitions not loaded in model properly'

# Here we represent fluxes as a pandas data frame, with forward and backwards fluxes as separate columns
# 
# **Note**: non-reversible reactions should always carry zero reverse flux!
# 
# Access example: fluxes['a_b'].reverse

fluxDict = {'a_b': [100, 0], 'b_d': [110, 50], 'b_ec': [20, 0], 'bc_de': [20, 0], 'd_f': [80, 0]}
fluxes = bayflux.FluxVector(model, fluxDict)

# define output EMU we want to simulate
F = model.metabolites.get_by_id('F')
outputEMU = bayflux.EMU(F,[1,2,3])

# test the ability to compare an EMU to a non-EMU object
def test_emu_equality():
    assert outputEMU != 1.0, 'comparison between EMU and float should be false'

# Perform a test to find all EMU reactions producing metabolite F
result = bayflux.findProducingEMUTransitions(outputEMU)

def test_findProducingEMUTransitions():
    assert len(result) == 1, 'precursor transitions to F not length 1 as expected'
    assert len(result[0].reactants) == 1, 'precursor EMUs to F not length 1 as expected'
    assert hash(result[0].reactants[0]) == hash(bayflux.EMU(m['D'],[1,2,3])), 'precursor EMU to F not D[1, 2, 3] as expected'

# perform test full EMU decomposition of metabolite F precursors
fullDecomposition = bayflux.emuDecomposition([outputEMU])

# Split apart EMU reactions by product size
transitionsBySize = bayflux.splitBySize(fullDecomposition)

def test_emuDecomposition():
    assert len(fullDecomposition) == 18, 'unexpected number of EMU reactions in the F[1, 2, 3] decomposition'
    assert sorted(transitionsBySize.keys()) == [1, 2, 3], 'unexpected or missing EMU sizes in the F[1, 2, 3] decomposition'
    return transitionsBySize

# Test ability to merge duplicates in the transition list
# by duplicating an element, and then confirming it gets removed
assert len(transitionsBySize[1]) == len(bayflux.mergeDuplicates(transitionsBySize[1])), 'length changed when merging duplicates in transition list without duplicates'
newTransitionList = transitionsBySize[1].copy()
newTransitionList.append(transitionsBySize[1][0])
assert len(newTransitionList) == len(transitionsBySize[1]) + 1
newTransitionList = bayflux.mergeDuplicates(newTransitionList)
assert len(newTransitionList) == len(transitionsBySize[1]) 

# Test identifying the internal and external EMUs for all size 1 EMU reactions, which is later used as the coordinates for the resulting matrices
transitionsBySize = test_emuDecomposition()
matrixCoords = bayflux.constructMatrixCoords(transitionsBySize[1])

def test_constructMatrixCoords():
    assert len(matrixCoords['externalEMUs']['hashes']) == 2, 'wrong number of external EMU hashes for EMU size 1 network'
    assert len(matrixCoords['externalEMUs']['objects']) == 2, 'wrong number of external EMU objects for EMU size 1 network'
    assert len(matrixCoords['externalEMUs']['text']) == 2, 'wrong number of external EMU text names for EMU size 1 network'

    assert len(matrixCoords['internalEMUs']['hashes']) == 5, 'wrong number of internal EMU hashes for EMU size 1 network'
    assert len(matrixCoords['internalEMUs']['objects']) == 5, 'wrong number of internal EMU objects for EMU size 1 network'
    assert len(matrixCoords['internalEMUs']['text']) == 5, 'wrong number of internal EMU text names for EMU size 1 network'

# Create example zero matrices for the length-1 EMU reactions
exampleMatrices = bayflux.constructZeroMatrices(matrixCoords)

def test_constructZeroMatrices():
    assert len(exampleMatrices) == 2, 'wrong number of zero matrices for EMU size 1'
    assert exampleMatrices[0].shape == (5,5), 'wrong dimensions for EMU size 1 A matrix'
    assert exampleMatrices[1].shape == (5,2), 'wrong dimensions for EMU size 1 B matrix'


# Example of compiling the size 1 EMU reactions into a set of matrix addition operations
operations = bayflux.compileEMUTransitionList(transitionsBySize[1], matrixCoords, model)

def test_compileEMUTransitionList():
    assert np.matrix(operations).shape == (18, 5), 'wrong shape/number of compiled operations for size 1 EMUs'

# demo creation of a mass distribution vectorfor a binary labeling vector

substrateLabelingDict = {
    m['A']: ((1.0, [0, 1, 0]),)
}

inputEMU = bayflux.EMU(m['A'], [1,2,3])
extractedLabeling = bayflux.extractSubstrateEMU(inputEMU, substrateLabelingDict)

def test_extractSubstrateEMU():
    assert extractedLabeling.tolist() == [0, 1, 0, 0], 'wrong extracted labeling for EMU A[1,2,3]'

# Run fully automated EMU simulation

# First let's decompose the EMU network, and compile it's structure into a list of matrix addition operations
# EMUs to simulate in list form
emusToSimulate = [bayflux.EMU(m['F'],[1,2,3])]


def test_simulateLabeling():
    # compile emus
    compiledData = bayflux.emuCompile(emusToSimulate, model, substrateLabelingDict)

    # Now we'll test the injection of a specific flux vector, in order to simulate the labeling of metabolite F
    resultsimulateLabeling = bayflux.simulateLabeling(compiledData, fluxes, substrateLabelingDict)

    Findex = compiledData[2]['matrixCoords']['internalEMUs']['hashes'].index(hash(emusToSimulate[0]))
    assert len(compiledData[2]['matrixCoords']['internalEMUs']['hashes']) == 3, 'wrong number of simulated size 3 EMUs'
    Flabeling = resultsimulateLabeling[3][Findex]

    assert Flabeling == pytest.approx([0.0001, 0.8008, 0.1983, 0.0009], abs=1e-4), 'Wrong predicted labeling for F metabolite'

def test_pruning_and_merging():
    # compile emus
    compiledData = bayflux.emuCompile(emusToSimulate, model, substrateLabelingDict, True, True)

    # Now we'll test the injection of a specific flux vector, in order to simulate the labeling of metabolite F
    resultsimulateLabeling = bayflux.simulateLabeling(compiledData, fluxes, substrateLabelingDict)

    Findex = compiledData[2]['matrixCoords']['internalEMUs']['hashes'].index(hash(emusToSimulate[0]))
    assert len(compiledData[2]['matrixCoords']['internalEMUs']['hashes']) == 2, 'wrong number of simulated size 3 EMUs'
    Flabeling = resultsimulateLabeling[3][Findex]

    assert Flabeling == pytest.approx([0.0001, 0.8008, 0.1983, 0.0009], abs=1e-4), 'Wrong predicted labeling for F metabolite'

def test_EMUTransition_repr():
    # test if repr(transition) prints correctly

    emu = bayflux.EMU(m['F'],[1,2,3])
    transition = bayflux.findProducingEMUTransitions(emu)[0]
    assert repr(transition) == '1.0*d_f: D[1, 2, 3]-> F[1, 2, 3]'
    transition.forward = False
    assert repr(transition) == '-1.0*d_f: D[1, 2, 3]-> F[1, 2, 3]'

def test_compileEMUTransitionList_exception():
    # test what happens if you have an invalid EMU in the transitionsList

    # define an invalid EMU
    transitionsBySize[1][0].reactants[0] = bayflux.EMU(m['F'],[1,2,3,4])

    with pytest.raises(Exception):
        bayflux.compileEMUTransitionList(transitionsBySize[1], matrixCoords, model)

def test_missing():
    # test what happens if we drop the transitions from a reaction
    F = model.metabolites.get_by_id('F')

    producingReaction = bayflux.findProducingReactions(F)[0][0]
    transitions = copy(producingReaction.transitions)
    producingReaction.transitions = [] 

    assert len(bayflux.findProducingReactions(F)) == 0, 'not skipping reactions missing transitions as expected'

    producingReaction.transitions = transitions
    assert len(bayflux.findProducingReactions(F)) == 1, 'not finding reactions with transitions as expected' 

def test_circular_reference():
    # test what happens when we create a network that has a 'side loop'
    # that is missing any carbon inputs, and feeds into the main model.
    # This seems to be a common problem with 13C MFA networks.

    # first copy the model
    badModel = model.copy()

    # create three new metabolites that will form a closed loop with no valid inputs
    G = cobra.Metabolite(
        'G',
        formula='C3',
        name='G',
        compartment='c')
    H = cobra.Metabolite(
        'H',
        formula='C3',
        name='H',
        compartment='c')
    I = cobra.Metabolite(
        'I',
        formula='C3',
        name='I',
        compartment='c')

    # create four reactions to define the G->H->I->G loop plus
    # connect it to the model with an H->B reaction
    g_h = cobra.Reaction('g_h')
    g_h.lower_bound = 0 
    g_h.upper_bound = 500
    g_h.add_metabolites({
        G: -1.0,
        H: 1.0
    })

    h_i = cobra.Reaction('h_i')
    h_i.lower_bound = 0 
    h_i.upper_bound = 500
    h_i.add_metabolites({
        H: -1.0,
        I: 1.0
    })

    i_g = cobra.Reaction('i_g')
    i_g.lower_bound = 0 
    i_g.upper_bound = 500
    i_g.add_metabolites({
        I: -1.0,
        G: 1.0
    })

    h_b = cobra.Reaction('h_b')
    h_b.lower_bound = 0 
    h_b.upper_bound = 500
    h_b.add_metabolites({
        H: -1.0,
        B: 1.0
    })

    badModel.add_reactions([g_h, h_i, i_g, h_b])

    # add atom transitions for the 'bad loop'
    m = {m.id:m for m in badModel.metabolites}
    badModel.reactions.g_h.transitions = [bayflux.AtomTransition(
            ((m['G'], [1,2,3]),), # reactant labels
            ((m['H'], [1,2,3]),) # product labels
        )]
    badModel.reactions.h_i.transitions = [bayflux.AtomTransition(
            ((m['H'], [1,2,3]),), # reactant labels
            ((m['I'], [1,2,3]),) # product labels
        )]
    badModel.reactions.i_g.transitions = [bayflux.AtomTransition(
            ((m['I'], [1,2,3]),), # reactant labels
            ((m['G'], [1,2,3]),) # product labels
        )]
    badModel.reactions.h_b.transitions = [bayflux.AtomTransition(
            ((m['H'], [1,2,3]),), # reactant labels
            ((m['B'], [1,2,3]),) # product labels
        )]

    # perform full compilation for F
    emusToSimulate = [bayflux.EMU(m['F'],[1,2,3])]
    substrateLabelingDict = {
        m['A']: ((1.0, [0, 1, 0]),)
    }

    # confirm that we get the circular loop error
    # which comes from identifying that the G->H->I->G loop has no carbon inputs
    with pytest.raises(AssertionError, match=r"^circular loop"):
        bayflux.emuCompile(emusToSimulate, badModel, substrateLabelingDict, prune=True)

