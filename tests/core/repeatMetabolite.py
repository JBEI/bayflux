import pytest
import cobra
import bayflux

''' This file tests repeated metabolite EMU computation
Example repeated metabolite reaction:
ACLS pyr + pyr --> alac-S + co2 cde + fgh : fgdhe + c 

Here we create this as a one reaction model and confirm it's correct labeling in both directions
'''

def testForwardRepeatM():
    # create a blank model
    model = cobra.Model('repeatedM')

    # define metabolites
    pyr = cobra.Metabolite(
        'pyr',
        name='pyr',
        compartment='c')
    alacS = cobra.Metabolite(
        'alac-S',
        name='alac-S',
        compartment='c')
    co2 = cobra.Metabolite(
        'co2',
        name='co2',
        compartment='c')

    # define reactions and add them to the model
    ACLS = cobra.Reaction('ACLS')
    ACLS.lower_bound = 0
    ACLS.upper_bound = 500
    ACLS.add_metabolites({
        pyr: -2.0,
        alacS: 1.0,
        co2: 1.0,
    })
    model.add_reactions([ACLS])


    # Now we convert this model to a bayflux ReactionNetwork, which inherets 'EnhancedReaction' objects from each cobra.Reaction, allowing us to add in atom transitions
    model = bayflux.ReactionNetwork(model)

    # Here we set the atom transitions for each reaction, and show an example of viewing an EnhancedReaction which reports atom transitions
    m = {m.id:m for m in model.metabolites}

    model.reactions.ACLS.transitions = [bayflux.AtomTransition(
            ((m['pyr'], [1,2,3]),(m['pyr'], [4,5,6]),), # reactant labels
            ((m['alac-S'], [4,5,2,6,3]),(m['co2'], [1]),) # product labels
        )]
    # Here we represent fluxes as a FluxVector object, with fluxes ordered to match the model variables
    # 
    # **Note**: non-reversible reactions should always carry zero reverse flux!
    fluxDict = {'ACLS': [1, 0]}
    fluxes = bayflux.FluxVector(model, fluxDict)

    # ### Now we define the substrate labeling vector, and test the ability to extract a mass distribution vector from this labeling
    # demo creation of a mass distribution vector for a binary labeling vector

    substrateLabelingDict = {
        m['pyr']:  ((1.0, [1, 0, 0]),),
    }

    # ### Run fully automated EMU simulation

    # #### First let's decompose the EMU network, and compile its structure into a list of matrix addition operations
    # EMUs to simulate in list form
    emusToSimulate = [bayflux.EMU(m['alac-S'],[0,1,2,3,4]), bayflux.EMU(m['co2'],[0])]

    compiledData = bayflux.emuCompile(emusToSimulate, model, substrateLabelingDict)
    results = bayflux.simulateLabeling(compiledData, fluxes, substrateLabelingDict)

    # confirm that CO2 is 100% labeled (from the first carbon of Pyr)
    assert results[1].tolist()[0] == [0.0, 1.0], 'incorrect computed labeling for repeated metabolite reaction'

    # confirm that alac-S is 100% 1 carbon labeled (from the first carbon of Pyr, coming from one of the two Pyrs)
    assert results[5].tolist()[0] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'incorrect computed labeling for repeated metabolite reaction'


def testBackwardsRepeatM():
    # ## Now let's try reversing this and checking the labeling running backwards

    # create a blank model
    model = cobra.Model('repeatedM')

    # define metabolites
    pyr = cobra.Metabolite(
        'pyr',
        name='pyr',
        compartment='c')
    alacS = cobra.Metabolite(
        'alac-S',
        name='alac-S',
        compartment='c')
    co2 = cobra.Metabolite(
        'co2',
        name='co2',
        compartment='c')

    # define reactions and add them to the model
    ACLS = cobra.Reaction('ACLS')
    ACLS.lower_bound = 0
    ACLS.upper_bound = 500
    ACLS.add_metabolites({
        pyr: 2.0,
        alacS: -1.0,
        co2: -1.0,
    })

    model.add_reactions([ACLS])

    model = bayflux.ReactionNetwork(model)

    fluxDict = {'ACLS': [1, 0]}
    fluxes = bayflux.FluxVector(model, fluxDict)

    # create dict of metabolites by name
    # we use this instead of directly using the metabolite IDs above, because the conversion to
    # a bayflux.ReactionNetwork created new metabolite objects
    m = {m.id:m for m in model.metabolites}

    model.reactions.ACLS.transitions = [bayflux.AtomTransition(
            ((m['alac-S'], [4,5,2,6,3]),(m['co2'], [1]),), # reactant labels
            ((m['pyr'], [1,2,3]),(m['pyr'], [4,5,6]),) # product labels
    )]

    pyr = model.metabolites.get_by_id('pyr')
    outputEMU = bayflux.EMU(pyr,[0,1,2])
    result = bayflux.findProducingEMUTransitions(outputEMU)

    # check that we get two transitions for Pyruvate from this one reaction
    assert len(result) == 2, 'wrong number of producting transitions for repeated metabolite'


    substrateLabelingDict = {
        m['alac-S']: ((1.0, [0, 0, 0, 0, 0]),),
        m['co2']: ((1.0, [1]),)
    }

    # EMUs to simulate in list form
    emusToSimulate = [bayflux.EMU(m['pyr'],[0,1,2])]

    compiledData = bayflux.emuCompile(emusToSimulate, model, substrateLabelingDict)

    results = bayflux.simulateLabeling(compiledData, fluxes, substrateLabelingDict)

    # confirm that pyruvate is 50% unlabeled (fraction from alac-S) and 50% 1 carbon (fraction from CO2)
    assert results[3].tolist()[0] == [0.5, 0.5, 0.0, 0.0], 'incorrect computed labeling for repeated metabolite reaction'

