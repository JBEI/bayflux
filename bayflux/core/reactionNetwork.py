import cobra
from copy import copy, deepcopy
import re
import csv
from string import ascii_lowercase, ascii_uppercase
import itertools
import pandas as pd
import numpy as np
import bayflux
from six import iteritems, string_types

class ReactionNetwork(cobra.Model):
    """ReactionNetwork contains a genome scale model with atom transitions.

    This inherits from a cobra.Model but reactions are replaced with
    EnhancedReaction to add in atom transitions.

    Args:
        cobrapyModel (cobra.Model): A cobra.Model to inherit from.

    """

    def __init__(self, cobrapyModel):
        super().__init__(cobrapyModel)
        
        # the following code is adapted from the cobrapy reaction copy
        # but instead makes EnhancedReactions instead of reactions
        
        do_not_copy_by_ref = {
            "metabolites",
            "reactions",
            "genes",
            "notes",
            "annotation",
            "groups",
        }
        for attr in cobrapyModel.__dict__:
            if attr not in do_not_copy_by_ref:
                self.__dict__[attr] = cobrapyModel.__dict__[attr]
        self.notes = deepcopy(cobrapyModel.notes)
        self.annotation = deepcopy(cobrapyModel.annotation)

        self.metabolites = cobra.DictList()
        do_not_copy_by_ref = {"_reaction", "_model"}
        for metabolite in cobrapyModel.metabolites:
            new_met = metabolite.__class__()
            for attr, value in metabolite.__dict__.items():
                if attr not in do_not_copy_by_ref:
                    new_met.__dict__[attr] = copy(
                        value) if attr == "formula" else value
            new_met._model = self
            self.metabolites.append(new_met)

        self.genes = cobra.DictList()
        for gene in cobrapyModel.genes:
            new_gene = gene.__class__(None)
            for attr, value in gene.__dict__.items():
                if attr not in do_not_copy_by_ref:
                    new_gene.__dict__[attr] = copy(
                        value) if attr == "formula" else value
            new_gene._model = self 
            self.genes.append(new_gene)
        
        self.reactions = cobra.DictList()
        do_not_copy_by_ref = {"_model", "_metabolites", "_genes"}
        for reaction in cobrapyModel.reactions:
            new_reaction = EnhancedReaction(reaction.__class__())
            for attr, value in reaction.__dict__.items():
                if attr not in do_not_copy_by_ref:
                    new_reaction.__dict__[attr] = copy(value)
            new_reaction._model = self
            self.reactions.append(new_reaction)
            # update awareness
            for metabolite, stoic in reaction._metabolites.items():
                new_met = self.metabolites.get_by_id(metabolite.id)
                new_reaction._metabolites[new_met] = stoic
                new_met._reaction.add(new_reaction)
            for gene in reaction._genes:
                new_gene = self.genes.get_by_id(gene.id)
                new_reaction._genes.add(new_gene)
                new_gene._reaction.add(new_reaction)
                
        self.groups = cobra.DictList()
        do_not_copy_by_ref = {"_model", "_members"}
        # Groups can be members of other groups. We initialize them first and
        # then update their members.
        for group in cobrapyModel.groups:
            new_group = group.__class__(group.id)
            for attr, value in iteritems(group.__dict__):
                if attr not in do_not_copy_by_ref:
                    new_group.__dict__[attr] = copy(value)
            new_group._model = self
            self.groups.append(new_group)
        for group in cobrapyModel.groups:
            new_group = self.groups.get_by_id(group.id)
            # update awareness, as in the reaction copies
            new_objects = []
            for member in group.members:
                if isinstance(member, cobra.core.Metabolite):
                    new_object = self.metabolites.get_by_id(member.id)
                elif isinstance(member, cobra.core.Reaction):
                    new_object = self.reactions.get_by_id(member.id)
                elif isinstance(member, cobra.core.Gene):
                    new_object = self.genes.get_by_id(member.id)
                elif isinstance(member, cobra.core.Group):
                    new_object = self.genes.get_by_id(member.id)
                else:
                    raise TypeError(
                        "The group member {!r} is unexpectedly not a "
                        "metabolite, reaction, gene, nor another "
                        "group.".format(member)
                    )
                new_objects.append(new_object)
            new_group.add_members(new_objects)

        try:
            self._solver = deepcopy(cobrapyModel.solver)
            # Cplex has an issue with deep copies
        except Exception:  # pragma: no cover
            self._solver = copy(cobrapyModel.solver)  # pragma: no cover

        # it doesn't make sense to retain the context of a copied model so
        # assign a new empty context
        self._contexts = list()

    def copy(self):
        """Provides a partial 'deepcopy' of the Model.  All of the Metabolite,
        Gene, and Reaction objects are created anew but in a faster fashion
        than deepcopy

        This code was taken verbatim from cobrapy, but modified
        to add EnhancedReactions.
        """
        new = ReactionNetwork(cobra.Model())
        do_not_copy_by_ref = {"metabolites", "reactions", "genes", "notes",
                              "annotation"}
        for attr in self.__dict__:
            if attr not in do_not_copy_by_ref:
                new.__dict__[attr] = self.__dict__[attr]
        new.notes = deepcopy(self.notes)
        new.annotation = deepcopy(self.annotation)

        new.metabolites = cobra.DictList()
        do_not_copy_by_ref = {"_reaction", "_model"}
        for metabolite in self.metabolites:
            new_met = metabolite.__class__()
            for attr, value in metabolite.__dict__.items():
                if attr not in do_not_copy_by_ref:
                    new_met.__dict__[attr] = copy(
                        value) if attr == "formula" else value
            new_met._model = new
            new.metabolites.append(new_met)

        new.genes = cobra.DictList()
        for gene in self.genes:
            new_gene = gene.__class__(None)
            for attr, value in gene.__dict__.items():
                if attr not in do_not_copy_by_ref:
                    new_gene.__dict__[attr] = copy(
                        value) if attr == "formula" else value
            new_gene._model = new
            new.genes.append(new_gene)

        new.reactions = cobra.DictList()
        do_not_copy_by_ref = {"_model", "_metabolites", "_genes"}
        for reaction in self.reactions:
            new_reaction = EnhancedReaction(cobra.Reaction())
            for attr, value in reaction.__dict__.items():
                if attr not in do_not_copy_by_ref:
                    new_reaction.__dict__[attr] = copy(value)
            new_reaction._model = new
            new.reactions.append(new_reaction)
            # update awareness
            for metabolite, stoic in reaction._metabolites.items():
                new_met = new.metabolites.get_by_id(metabolite.id)
                new_reaction._metabolites[new_met] = stoic
                new_met._reaction.add(new_reaction)
            for gene in reaction._genes:
                new_gene = new.genes.get_by_id(gene.id)
                new_reaction._genes.add(new_gene)
                new_gene._reaction.add(new_reaction)
            for transition in reaction.transitions:
                newReactants = tuple([(new.metabolites.get_by_id(r[0].id), r[1]) for r in transition.reactants])
                newProducts = tuple([(new.metabolites.get_by_id(r[0].id), r[1]) for r in transition.products])
                new_reaction.transitions.append(AtomTransition(newReactants, newProducts))
        try:
            new._solver = deepcopy(self.solver)
            # Cplex has an issue with deep copies
        except Exception:  # pragma: no cover
            new._solver = copy(self.solver)  # pragma: no cover

        # it doesn't make sense to retain the context of a copied model so
        # assign a new empty context
        new._contexts = list()

        return new

    def writeFluxConstraints(self, outputFileName, format='csv', exchangesOnly=True):
        # export flux constraints to a file
    
        assert format in {'csv'}, 'unsupported format'
        
        # decide which reactions to use
        if exchangesOnly:
            reactions = self.exchanges
        else:
            reactions = self.reactions
            
        # create data frame where index is the reaction id and the columns
        # are the lower and upper bounds on each flux respectively
        fluxBoundDF = pd.DataFrame(
            [(r.lower_bound, r.upper_bound) for r in self.exchanges], 
            dtype=np.float64, 
            index=[r.id for r in reactions], 
            columns=['lower_bound', 'upper_bound'])
        
        # write it!
        fluxBoundDF.to_csv(outputFileName)

    def readFluxConstraints(self, inputFileName, format='csv'):
    # Import and apply exchange flux constraints from file
        
        assert format in {'csv'}, 'unsupported format'
        
        # read the file
        fluxBoundDF = pd.read_csv(
            inputFileName, 
            header=0, 
            index_col=0, 
            dtype={'lower_bound': np.float64, 'upper_bound': np.float64},
            float_precision='round_trip')
        
        # iterate over rows and apply them
        for row in fluxBoundDF.iterrows():
            
            # get series for this row and unpack variables
            rowSeries = row[1]
            reactionId = rowSeries.name
            lowerBound = rowSeries[0]
            upperBound = rowSeries[1]
            
            # check that lower bound is leq upper bound
            assert lowerBound <= upperBound, 'error applying lower bound larger than upper bound'
            
            # check that reaction is in the self
            assert reactionId in self.reactions, 'error applying flux for reaction missing from self: ' + str(reactionId)
            
            # apply flux bounds to self
            reaction = self.reactions.get_by_id(reactionId)
            reaction.lower_bound = lowerBound
            reaction.upper_bound = upperBound

    def writeAtomTransitions(self, outputFileName, format='bayflux'):
        
        assert format in {'bayflux'}, 'unsupported format'
        # note: bayflux format is the same as the jqmm1 format, except 
        # symmetric metabolites are encoded as multiple transitions, instead of in parentheses
        
        transitions = []
        
        for r in self.reactions:
            
            # skip reactions with no transition
            if len(r.transitions) < 1:
                next
            
            # if multiple transitions, apply them
            for transition in r.transitions:
                transitionString = repr(transition)
                
                # use the reversible syntax if the reaction is reversible
                if r.lower_bound < 0.0:
                    transitionString = transitionString.replace('-->', '<==>', 1)
                
                # split the transition string apart by tabs, and append to list
                transitionList = transitionString.split('\t')
                transitions.append([r.id] + transitionList)

        # create data frame where index is the reaction id and the columns
        # are the lower and upper bounds on each flux respectively
        transitionsDF = pd.DataFrame(
            transitions, 
            dtype=str)
        
        # write it!
        transitionsDF.to_csv(outputFileName, sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

    def readAtomTransitions(self, inputFileName, format='bayflux'):

        assert format in {'bayflux'}, 'unsupported format'
        # note: bayflux format is the same as the jqmm1 format, except 
        # symmetric metabolites are encoded as multiple transitions, instead of in parentheses
        
        # create a dict for looking up self metabolites by name
        m = {m.id:m for m in self.metabolites}
        
        # parse the tab separated file
        transitionsDF = pd.read_csv(
            inputFileName, 
            header=None, 
            sep='\t',
            index_col=0, 
            comment='#',
            names=[0,1,2],
            dtype=str)
        
        # iterate over transitions, and apply them to the self
        for row in transitionsDF.iterrows():
            
            # parse out reaction name and confirm it occurs in the self
            rowSeries = row[1]
            reactionName = rowSeries.name.strip()
            assert reactionName in self.reactions, 'reaction ' + reactionName + ' in transitions is missing from self'
            reaction = self.reactions.get_by_id(reactionName)
            
            # get metabolite and indices strings
            assert len(rowSeries) == 2, 'incorrectly formatted transition for reaction ' + reactionName
            
            # parse out metabolites and products
            metabolites = rowSeries.iloc[0]
            reactants, products = re.split('\s+<==>\s+|\s+-->\s+', metabolites)
            reactantList = [i.strip() for i in re.split('\s+\+\s+', reactants)]
            productList = [i.strip() for i in re.split('\s+\+\s+', products)]
            assert len(set(reactantList).intersection([m.id for m in reaction.reactants])) == len(set(reactantList)), 'reactant missing from reaction ' + str(row)
            assert len(set(productList).intersection([m.id for m in reaction.products])) == len(set(productList)), 'product missing from reaction ' + str(row)
            
            # parse out atom mapping indices
            indices = rowSeries.iloc[1]
            reactantIndices, productIndices = re.split('\s+:\s+', indices)
            reactantIndicesList = re.split('\s+\+\s+', reactantIndices)
            reactantIndicesList = [i.strip() for i in reactantIndicesList]
            assert len(reactantIndicesList) == len(reactantList), 'differing number of reactant names and indices ' + str(row)
            productIndicesList = re.split('\s+\+\s+', productIndices)
            productIndicesList = [i.strip() for i in productIndicesList]
            assert len(productIndicesList) == len(productList), 'differing number of product names and indices ' + str(row)
            
            # confirm that indices are balanced
            assert sum([len(s) for s in productIndicesList]) == sum([len(s) for s in reactantIndicesList]), 'differing number of indices for reactants and products ' + str(row) + str(productIndicesList) + ':' + str(reactantIndicesList)
            
            # confirm that indices are unique
            productIndexString = ''.join(productIndicesList)
            assert len(set(productIndexString)) == len(productIndexString), 'duplicated indices in transition ' + str(row)
            reactantIndexString = ''.join(reactantIndicesList)
            assert len(set(reactantIndexString)) == len(reactantIndexString), 'duplicated indices in transition ' + str(row)
            
            # confirm that indicies are the same left and right
            assert len(set(productIndexString).intersection(reactantIndexString)) == len(set(productIndexString)), 'differing number of reactant and product indicies in transition ' + str(row)
            
            # create the new transition object
            newTransition = bayflux.AtomTransition(
                tuple(zip([m[mname] for mname in reactantList], [[ord(c) for c in s] for s in reactantIndicesList])), # reactant labels
                tuple(zip([m[mname] for mname in productList], [[ord(c) for c in s] for s in productIndicesList])) # product labels
             )
            
            # add transition to self
            self.reactions.get_by_id(reactionName).transitions.append(newTransition)

class EnhancedReaction(cobra.Reaction):
    """like a cobra.Reaction, but with atom transitions.

    See the attributes for cobra.Reaction in cobrapy

    Args:
        cobrapyReaction (cobra.Reaction): a cobra.Reaction object to 
            inherit from.
        transitions (list, optional): a list of AtomTransition objects.

    Attributes:
        transitions (list, optional): a list of AtomTransition objects.

    """

    # like a reaction, but with atom transitions
    
    def __init__(self, cobrapyReaction, transitions=None):
        super().__init__(cobrapyReaction)
        if not transitions:
            transitions = []
        self.transitions = transitions
        
    def _repr_html_(self):
        oldHtml = super(EnhancedReaction, self)._repr_html_()
        newHtml = re.sub('</table>', '', oldHtml)
        for transition in self.transitions:
            newHtml += """
                    <tr>
                        <td><strong>Atom transition</strong></td><td>{transition}</td>
                    </tr>
                """.format(transition=repr(transition))
            newHtml += """
                </table>
                """
        return newHtml

class AtomTransition():
    """Class for storing atom transitions associated with a reaction.

    Overview
    reactant and product transitions are
    stored as a tuple of two value tuples where the first value of each
    two value tuple is a cobra.Metabolite
    and the second value is a tuple of integers where the position
    in the tuple represents an atom (in left to right order)
    from the canonical SMILES representation of the metabolite,
    and the values in the tuples are integers, whereby the same
    atom in a product and reactant have the same integer label.
    Specific values of the integers are arbitrary.

    Special cases:

    Reactions with multiple atom transitions
    MetaCyc reaction SUCCCOASYN-RXN is an example
    with multiple atom transitions. We accommodate this
    by associating multiple atom transition objects with the
    same EnhancedReaction.

    Equivalent atoms
    Biochemically indistinguishable atoms should be represented
    with the same integer value in the atom transition tuple. This
    will be used to generate a set of equivalent EMUs. 

    Args:
        reactants (tuple): A tuple of two valued tuples as described above. 
        products (tuple): A tuple of two valued tuples as described above. 

    Attributes:
        reactants (tuple): A tuple of two valued tuples as described above. 
        products (tuple): A tuple of two valued tuples as described above. 

    """

    def __init__(self, reactants, products):
        assert len(reactants) > 0, 'No reactants in transition'
        self.reactants = reactants

        assert len(products) > 0, 'No products in transition'
        self.products = products

        # get list of indices for products and reactants
        allProductIndices = sorted(itertools.chain.from_iterable([m[1] for m in self.products]))
        allReactantIndices = sorted(itertools.chain.from_iterable([m[1] for m in self.reactants]))

        assert allProductIndices == allReactantIndices, 'Transition is unbalanced'
            
    def reverse(self):
        return AtomTransition(reactants=self.products, products=self.reactants)

    def __repr__(self):
        # print a jQMM v. 1 style transition string
        # Eventually move this into the EnhancedReaction class
        # so it knows about reversibility and multiple transitions
        reaction = ''
        transitions = ''
        alphabetIter = iter(list(ascii_lowercase + ascii_uppercase + \
            ''.join([chr(i) for i in range(255,600)])))
        alphabetLookup = {} # store lookup from integers to alphabet
        for i, reactantTuple in enumerate(self.reactants):
            metabolite, transitionValues = reactantTuple
            if i != 0:
                reaction += ' + '
                transitions += ' + '
            reaction += metabolite.id
            for transitionValue in transitionValues:
                uniqueLetter = alphabetIter.__next__()
                alphabetLookup[transitionValue] = uniqueLetter
                transitions += uniqueLetter
        reaction += ' --> '
        transitions += ' : '
        for i, productTuple in enumerate(self.products):
            metabolite, transitionValues = productTuple
            if i != 0:
                reaction += ' + '
                transitions += ' + '
            reaction += metabolite.id
            for transitionValue in transitionValues:
                transitions += alphabetLookup[transitionValue]

        return reaction + '\t' + transitions

