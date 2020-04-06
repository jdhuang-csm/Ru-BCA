import os
import numpy as np
import pymatgen as mg
from pymatgen.ext.matproj import MPRester
import collections
from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.composition import ValenceOrbital, CohesiveEnergy, ElementProperty, BandCenter, MolecularOrbitals
from matminer.utils.data import MagpieData
import pandas as pd

from bca_plotting import get_coords_from_comp

"""Lookups for Ba, Ca, and Al"""
oxides = {'Ba':'BaO','Ca':'CaO','Al':'Al2O3'}
nitrides = {'Ba':'Ba3N2','Ca':'Ca3N2','Al':'AlN'}
hydrides = {'Ba':'BaH2','Ca':'CaH2','Al':'AlH3'}
# Elemental work function (eV) - from https://public.wsu.edu/~pchemlab/documents/Work-functionvalues.pdf
# Al value is average of 100,110,111 planes; Ba and Ca values are for polycrystalline
work_function = {'Ba':2.52,'Ca':2.87,'Al':4.17}

"""Load elemental electrical conductivity data"""
elec_conductivity_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),'ElementalElectricalConductivity.txt'),sep='\t',skipfooter=1,engine='python')
elec_conductivity = dict(zip(elec_conductivity_df['Symbol'],elec_conductivity_df['Electrical Conductivity (S/cm)']))

class MatProjCalc:
	def __init__(self,oxide_dict={}):
		#dict to store MX bond energies after calculation. Avoid repeated lookups in MP
		self.calc_MX_bond_energy = {} 
		#dict to store formation enthalpies after looking up
		self.fH_dict = {
				('Ce','gas','exp'):(417.1,'Formation enthalpy for Ce in gas phase includes exp data from phases: gas') #correction to MP entry: fH for Ce gas is negative in MP
					}
		self.mp = MPRester(os.environ['MATPROJ_API_KEY'])
		print("Created MatProjCalc instance")
		
	@property
	def common_anions(self):
		"""List of common anions"""
		return ['N','P','O','S','F','Cl','Br','I']
		
	@property
	def dissocation_energy(self):
		"""
		Bond dissociation energies for gases at 298K in kJ/mol
		Source: https://labs.chem.ucsb.edu/zakarian/armen/11---bonddissociationenergy.pdf
		"""
		return dict(N=945.33,P=490,O=498.34,S=429,F=156.9,Cl=242.58,Br=193.87,I=152.549,H=436.002)
		
	@property
	def mn_combos(self):
		"""
		Possible m-n pairs (m,n up to 4)
		"""
		return [(1,1),(1,2),(1,3),(1,4),(2,1),(2,3),(3,1),(3,2),(3,4),(4,1)]
		
	def possible_ionic_formulas(self,metal,anion,metal_ox_lim=None,anion_ox_state=None):
		"""
		Get possible binary ionic compound formulas for metal-anion pair
		
		Parameters:
		-----------
		metal: metal element symbol
		anion: anion element symbol
		metal_ox_lim: tuple of metal oxidation state limits (min, max)
		anion_ox_state: anion oxidation state. If None, will attempt to find the common oxidation state for the anion 
		"""
		#get common oxidation state for anion
		if anion_ox_state is None:
			anion_ox_state = [ox for ox in mg.Element(anion).common_oxidation_states if ox < 0]
			if len(anion_ox_state) > 1:
				raise Exception(f"Multiple common oxidation states for {anion}. Please specify anion_ox_state")
			else:
				anion_ox_state = anion_ox_state[0]
				
		if metal_ox_lim is None:
			metal_ox_lim = [0,np.inf]
		
		return [f'{metal}{m}{anion}{n}' for m,n in self.mn_combos if m/n <= -anion_ox_state and metal_ox_lim[0] <= -anion_ox_state*n/m <= metal_ox_lim[1]]
		
	def get_fH(self,formula, phase='solid', data_type='exp',silent=True):
		"""
		Get average experimental formation enthalpy for formula and phase
		
		Parameters:
		-----------
		formula: chemical formula string
		phase: phase string. Can be 'solid', 'liquid', 'gas', or a specific solid phase (e.g. 'monoclinic'). If 'solid', returns average across all solid phases
		"""
		#first check for corrected/saved data in fH_dict
		try:
			fH,msg = self.fH_dict[(formula,phase,data_type)]
			if silent==False:
				#print('already calculated')
				print(msg)
		#if no entry exists, look up in MP
		except KeyError:
			results = self.mp.get_data(formula,data_type=data_type)
			if data_type=='exp':
				#results = self.mp.get_exp_thermo_data(formula)
				if phase=='solid':
					phase_results = [r for r in results if r.type=='fH' and r.phaseinfo not in ('liquid','gas')]
				else:
					phase_results = [r for r in results if r.type=='fH' and r.phaseinfo==phase]
				phases = np.unique([r.phaseinfo for r in phase_results])
				fH = [r.value for r in phase_results]
				
			elif data_type=='vasp':
				if phase in ('liquid','gas'):
					raise ValueError('VASP data only valid for solid phases')
				elif phase=='solid':
					#get entry with lowest energy above hull
					srt_results = sorted(results,key=lambda x: x['e_above_hull'])
					phase_results = srt_results[0:1]
				else:
					phase_results = [r for r in results if r['spacegroup']['crystal_system']==phase]
				phases = np.unique([r['spacegroup']['crystal_system'] for r in phase_results])
				n_atoms = mg.Composition(formula).num_atoms
				#DFT formation energies given in eV per atom - need to convert to kJ/mol
				fH = [r['formation_energy_per_atom']*n_atoms*96.485 for r in phase_results]
				
			if len(fH)==0:
				raise LookupError('No {} data for {} in {} phase'.format(data_type,formula,phase))
			maxdiff = np.max(fH) - np.min(fH)
			if maxdiff > 15:
				warnings.warn('Max discrepancy of {} in formation enthalpies for {} exceeds limit'.format(maxdiff,formula))
			fH = np.mean(fH)
			
			msg = 'Formation enthalpy for {} in {} phase includes {} data from phases: {}'.format(formula,phase,data_type,', '.join(phases))
			if silent==False:
				print(msg)
			
			#store value and info message for future lookup
			self.fH_dict[(formula,phase,data_type)] = (fH,msg)
			
		return fH

	def ionic_formula_from_ox_state(self,metal,anion,metal_ox_state,anion_ox_state=None,return_mn=False):
		"""
		Get binary ionic compound formula with reduced integer units based on oxidation state
		
		Parameters:
		-----------
		metal: metal element symbol
		anion: anion element symbol
		metal_ox_state: metal oxidation state
		anion_ox_state: anion oxidation state. If None, will attempt to find the common oxidation state for the anion
		return_mn: if True, return formula units for metal (m) and anion (n)
		
		Returns: chemical formula string MmXn, and m, n if return_mn=True
		"""
		#get common oxidation state for anion
		if anion_ox_state is None:
			anion_ox_state = [ox for ox in mg.Element(anion).common_oxidation_states if ox < 0]
			if len(anion_ox_state) > 1:
				raise Exception(f"Multiple common oxidation states for {anion}. Please specify anion_ox_state")
			else:
				anion_ox_state = anion_ox_state[0]
				
		#formula MmXn
		deno = gcd(metal_ox_state,-anion_ox_state)
		m = -anion_ox_state/deno
		n = metal_ox_state/deno
		formula = '{}{}{}{}'.format(metal,m,anion,n)
		if return_mn==False:
			return formula
		else:
			return formula, m, n
			
	def ox_states_from_binary_formula(self,formula,anion=None,anion_ox_state=None):
		"""
		Determine oxidation states from binary formula.
		Could also use mg.Composition.oxi_state_guesses(), but the logic used is more complex.

		Args:
			formula: chemical formula
			anion: Element symbol of anion. If None, search for common anion
			anion_ox_state: oxidation state of anion. If None, assume common oxidation state
		"""
		comp = mg.Composition(formula)
		if len(comp.elements) != 2:
			raise ValueError('Formula must be binary')
		# determine anion
		if anion is None:
			anion = np.intersect1d([e.name for e in comp.elements],self.common_anions)
			if len(anion) > 1:
				raise ValueError('Found multiple possible anions in formula. Please specify anion')
			elif len(anion)==0:
				raise ValueError('No common anions found in formula. Please specify anion')
			else:
				anion = anion[0]
		metal = np.setdiff1d(comp.elements,mg.Element(anion))[0].name
			
		#get common oxidation state for anion
		if anion_ox_state is None:
			anion_ox_state = [ox for ox in mg.Element(anion).common_oxidation_states if ox < 0]
			if len(anion_ox_state) > 1:
				raise Exception(f"Multiple common oxidation states for {anion}. Please specify anion_ox_state")
			else:
				anion_ox_state = anion_ox_state[0]
				
		metal_ox_state = -comp.get(anion)*anion_ox_state/comp.get(metal)
		
		return {metal:metal_ox_state,anion:anion_ox_state}

	def MX_bond_energy(self,formula,data_type='exp',ordered_formula=False,silent=True):
		"""
		Get metal-anion bond energy per mole of metal for binary ionic compound
		
		Parameters:
		-----------
		formula: chemical formula string
		ordered_formula: if true, assume that first element in formula is metal, and second is anion (i.e. MmXn)
		"""
		
		comp = mg.Composition(formula)
		formula = comp.reduced_formula
		try:
			#look up compound if already calculated
			abe,msg = self.calc_MX_bond_energy[(formula,data_type)]
			if silent==False:
				#print('already calculated')
				print(msg)
		except KeyError:
			if len(comp.elements) != 2:
				raise Exception("Formula is not a binary compound")
				
			if ordered_formula is False:
				anions = [el.name for el in comp.elements if el.name in self.common_anions]
				if len(anions) == 0:
					raise Exception('No common anions found in formula. Use ordered formula to indicate metal and anion')
				elif len(anions) > 1:
					raise Exception('Multiple anions found in formula.  Use ordered formula to indicate metal and anion')
				else:
					anion = anions[0]
				metal = [el.name for el in comp.elements if el.name!=anion][0]
			elif ordered_formula is True:
				metal = comp.elements[0].name
				anion = comp.elements[1].name
				
			m = comp.get_el_amt_dict()[metal]
			n = comp.get_el_amt_dict()[anion]
				
			fH = self.get_fH(formula,data_type=data_type,silent=silent) #oxide formation enthalpy
			H_sub = self.get_fH(metal, phase='gas',silent=silent) #metal sublimation enthalpy - must be exp data (no vasp data for gas)
			#look up info messages from get_fH to store in dict
			msg = self.fH_dict[formula,'solid',data_type][1] + '\n'
			msg += self.fH_dict[metal,'gas','exp'][1]
			DX2 = self.dissocation_energy[anion] #anion dissociation energy
			abe = (fH - m*H_sub - (n/2)*DX2)/m #M-O bond energy per mole of M
			self.calc_MX_bond_energy[(formula,data_type)] = (abe,msg)
		return abe
		
	def citations(self):
		"""Cite Materials Project, Materials API, and pymatgen"""
		return [
				"@article{Jain2013,"
					"author = {Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and Persson, Kristin a.},"
					"doi = {10.1063/1.4812323},"
					"issn = {2166532X},"
					"journal = {APL Materials},"
					"number = {1},"
					"pages = {011002},"
					"title = {{The Materials Project: A materials genome approach to accelerating materials innovation}},"
					"url = {http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\&Agg=doi},"
					"volume = {1},"
					"year = {2013}"
				"}",
				"@article{Ong_2015,"
					"doi = {10.1016/j.commatsci.2014.10.037},"
					"url = {http://dx.doi.org/10.1016/j.commatsci.2014.10.037},"
					"year = 2015,"
					"month = {feb},"
					"publisher = {Elsevier {BV}},"
					"volume = {97},"
					"pages = {209--215},"
					"author = {Shyue Ping Ong and Shreyas Cholia and Anubhav Jain and Miriam Brafman and Dan Gunter and Gerbrand Ceder and Kristin A. Persson},"
					"title = {The Materials Application Programming Interface ({API}): A simple, flexible and efficient {API} for materials data based on {REpresentational} State Transfer ({REST}) principles},"
					"journal = {Computational Materials Science}"
				"}",
				"@article{Ong2012b,"
					"author = {Ong, Shyue Ping and Richards, William Davidson and Jain, Anubhav and Hautier, Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and Chevrier, Vincent L. and Persson, Kristin A. and Ceder, Gerbrand},"
					"doi = {10.1016/j.commatsci.2012.10.028},"
					"file = {:Users/shyue/Mendeley Desktop/Ong et al/Computational Materials Science/2013 - Ong et al. - Python Materials Genomics (pymatgen) A robust, open-source python library for materials analysis.pdf:pdf;:Users/shyue/Mendeley Desktop/Ong et al/Computational Materials Science/2013 - Ong et al. - Python Materials Genomics (pymatgen) A robust, open-source python library for materials analysis(2).pdf:pdf},"
					"issn = {09270256},"
					"journal = {Computational Materials Science},"
					"month = feb,"
					"pages = {314--319},"
					"title = {{Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis}},"
					"url = {http://linkinghub.elsevier.com/retrieve/pii/S0927025612006295},"
					"volume = {68},"
					"year = {2013}"
				"}"
				]

#create MatProjCalc instance to store fetched data/calculations
mpcalc = MatProjCalc()

mpcalc = MatProjCalc()
# oxide formation enthalpies
oxide_Hf = {M:mpcalc.get_fH(MO) for M,MO in oxides.items()}
#bond energies per mole of metal M
MO_bond_energy = {M:mpcalc.MX_bond_energy(MO) for M,MO in oxides.items()}
MN_bond_energy = {M:mpcalc.MX_bond_energy(MN) for M,MN in nitrides.items()}
MH_bond_energy = {M:mpcalc.MX_bond_energy(MH,ordered_formula=True) for M,MH in hydrides.items()}
#bond energy delta per mole of M
ON_BondEnergyDelta = {M:MN_bond_energy[M] - MOBE  for M, MOBE in MO_bond_energy.items()}
OH_BondEnergyDelta = {M:MH_bond_energy[M] - MOBE  for M, MOBE in MO_bond_energy.items()}
NH_BondEnergyDelta = {M:MH_bond_energy[M] - MNBE  for M, MNBE in MN_bond_energy.items()}

oxidation_state = {'Ba':2,'Ca':2,'Al':3}

metal_lookups = {}
for m in ['Ba','Ca','Al']:
	metal_lookups[m] = {'work_function':work_function[m],
						'MO_BondEnergy':MO_bond_energy[m],
						'MN_BondEnergy':MN_bond_energy[m],
						'MH_BondEnergy':MH_bond_energy[m],
						'oxidation_state':oxidation_state[m],
						'ON_BondEnergyDelta':ON_BondEnergyDelta[m],
						'OH_BondEnergyDelta':OH_BondEnergyDelta[m],
						'NH_BondEnergyDelta':NH_BondEnergyDelta[m],
						'oxide_Hf':oxide_Hf[m]
					   }
					   
					   
class AtomicOrbitalsMod(BaseFeaturizer):
	"""
	*Modified from matminer class to handle cases where LUMO is None*
	Determine HOMO/LUMO features based on a composition.
	The highest occupied molecular orbital (HOMO) and lowest unoccupied
	molecular orbital (LUMO) are estiated from the atomic orbital energies
	of the composition. The atomic orbital energies are from NIST:
	https://www.nist.gov/pml/data/atomic-reference-data-electronic-structure-calculations
	Warning:
	For compositions with inter-species fractions greater than 10,000 (e.g.
	dilute alloys such as FeC0.00001) the composition will be truncated (to Fe
	in this example). In such extreme cases, the truncation likely reflects the
	true physics of the situation (i.e. that the dilute element does not
	significantly contribute orbital character to the band structure), but the
	user should be aware of this behavior.
	"""

	def featurize(self, comp):
		"""
		Args:
			comp: (Composition)
				pymatgen Composition object
		Returns:
			HOMO_character: (str) orbital symbol ('s', 'p', 'd', or 'f')
			HOMO_element: (str) symbol of element for HOMO
			HOMO_energy: (float in eV) absolute energy of HOMO
			LUMO_character: (str) orbital symbol ('s', 'p', 'd', or 'f')
			LUMO_element: (str) symbol of element for LUMO
			LUMO_energy: (float in eV) absolute energy of LUMO
			gap_AO: (float in eV)
				the estimated bandgap from HOMO and LUMO energeis
		"""

		integer_comp, factor = comp.get_integer_formula_and_factor()

		# warning message if composition is dilute and truncated
		if not (len(mg.Composition(comp).elements) ==
				len(mg.Composition(integer_comp).elements)):
			warn('AtomicOrbitals: {} truncated to {}'.format(comp,
															 integer_comp))

		homo_lumo = MolecularOrbitals(integer_comp).band_edges

		feat = collections.OrderedDict()
		
		for edge in ['HOMO', 'LUMO']:
			if homo_lumo[edge] is not None:
				feat['{}_character'.format(edge)] = homo_lumo[edge][1][-1]
				feat['{}_element'.format(edge)] = homo_lumo[edge][0]
				feat['{}_energy'.format(edge)] = homo_lumo[edge][2]
			else:
				#if LUMO is None
				feat['{}_character'.format(edge)] = 'na'
				feat['{}_element'.format(edge)] = 'na'
				#unclear what this value should be. Arbitrarily set to 0. Don't want NaN for modeling
				feat['{}_energy'.format(edge)] = 0 
				
		feat['gap_AO'] = feat['LUMO_energy'] - feat['HOMO_energy']

		return list(feat.values())

	def feature_labels(self):
		feat = []
		for edge in ['HOMO', 'LUMO']:
			feat.extend(['{}_character'.format(edge),
						 '{}_element'.format(edge),
						 '{}_energy'.format(edge)])
		feat.append("gap_AO")
		return feat

	def citations(self):
		return [
			"@article{PhysRevA.55.191,"
			"title = {Local-density-functional calculations of the energy of atoms},"
			"author = {Kotochigova, Svetlana and Levine, Zachary H. and Shirley, "
			"Eric L. and Stiles, M. D. and Clark, Charles W.},"
			"journal = {Phys. Rev. A}, volume = {55}, issue = {1}, pages = {191--199},"
			"year = {1997}, month = {Jan}, publisher = {American Physical Society},"
			"doi = {10.1103/PhysRevA.55.191}, "
			"url = {https://link.aps.org/doi/10.1103/PhysRevA.55.191}}"]
			
	def implementors(self):
		return ['Maxwell Dylla', 'Anubhav Jain']
					   
class ValenceOrbitalEnergy(BaseFeaturizer):
	def __init__(self):
		self.element_props = {}
		self.MagpieData = MagpieData()
	
	def get_element_props(self,el):
		try:
			props = self.element_props[el]
		except KeyError:
			subshells = 'spdf'
			n_elec = {sub:self.MagpieData.get_elemental_property(el,f'N{sub}Valence') for sub in subshells}
			orbitals = sorted(el.atomic_orbitals.keys())[::-1]
			#look up valence orbital for subshell
			orbital_func = lambda x: '{}{}'.format(max([orb[0] for orb in orbitals if orb[1]==x]),x)
			#get valence orbital energy for subshell
			energy_func = lambda x: el.atomic_orbitals[orbital_func(x)]
			props = {x:{'n_elec':n_elec[x],'energy':energy_func(x),'shell':orbital_func(x)[0]} for x in subshells if n_elec[x]>0}
			self.element_props[el] = props
		
		return props
		
	def featurize(self,comp):
		tot_energy = 0
		tot_elec = 0
		for el in comp.elements:
			props = self.get_element_props(el)
			tot_energy += comp[el]*sum([v['energy']*v['n_elec'] for v in props.values()])
			tot_elec += comp[el]*sum([v['n_elec'] for v in props.values()])
			
		return [tot_energy/tot_elec]
	
	def feature_labels(self):
		return ['MeanValenceEnergy']	

	def citations(self):
		return [
			"@article{Ward2016,"
			"archivePrefix = {arXiv},"
			"arxivId = {1606.09551},"
			"author = {Ward, Logan and Agrawal, Ankit and Choudhary, Alok and Wolverton, Christopher},"
			"doi = {10.1038/npjcompumats.2016.28},"
			"eprint = {1606.09551},"
			"isbn = {2057-3960},"
			"issn = {20573960},"
			"journal = {npj Computational Materials},"
			"number = {June},"
			"pages = {1--7},"
			"title = {{A general-purpose machine learning framework for predicting properties of inorganic materials}},"
			"volume = {2},"
			"year = {2016}"
			"}"]

					   
class BCA():
	def __init__(self,composition,radius_type='ionic_radius',normalize_formula=False):
		self.cations = [el.name for el in composition.elements if el.name!='O']
		self.radius_type = radius_type
		
		if normalize_formula==True:
			#scale to single total unit of cations
			tot_cat_amt = sum([composition[c] for c in self.cations])
			composition = mg.Composition({el:amt/tot_cat_amt for el,amt in composition.get_el_amt_dict().items()})
			
		self.composition = composition
		self.metal_composition = mg.Composition({c:self.composition[c] for c in self.cations})
		
		#checks
		if len(self.cations)==0:
			raise Exception('No cations in composition')
		if self.composition['O']!=self.composition['Ba'] + self.composition['Ca'] + self.composition['Al']*3/2:
			raise Exception('Oxygen amount does not match BaO, CaO, Al2O3 stoichiometry')
		if self.radius_type not in ('crystal_radius','ionic_radius'):
			raise Exception(f'Invalid radius type {self.radius_type}. Options are crystal_radius and ionic_radius')
	
	@property
	def tot_cat_amt(self):
		return sum([self.composition[c] for c in self.cations])
	
	def metal_mean_func(self,func):
		"""
		Weighted average of function func across metals in composition
		"""
		weights=list(self.metal_composition.get_el_amt_dict().values())
		return np.average([func(el) for el in self.metal_composition.elements],weights=weights)
	
	def metal_std_func(self,func,mean=None):
		"""
		Standard deviation of function func across metals in composition
		
		Args:
			func: function to average. Takes pymatgen Element as input
			mean: mean of function if already known. If None, calculated
		"""
		if mean is None:
			mean = self.metal_mean_func(func)
		return self.metal_mean_func(lambda el: (func(el) - mean)**2)**0.5
		
	# def is_in_phase_triangle(self):
		# xc,xa,xb = get_coords_from_comp(self.composition)
		# # One boundary specified in each line of if conditions: 1: BA to BCA723; 2: BA to C; 3: BCA723 to C
		# if xb <= 0.5*(1+xc) and xa >= 0.5 - 1.5*xc \
		# and xb >= 0.5*(1-xc) and xa <= 0.5*(1-xc) \
		# and xb <= (7/12)*(1-(6/5)*(xc-1/6)) and xa >= 0.25*(1-(6/5)*(xc-1/6)):
			# in_triangle = 1
		# else:
			# in_triangle = 0
		# return in_triangle
	
	def featurize(self):
		features = {}
		features['MO_ratio'] = self.tot_cat_amt/self.composition['O']
		#metal mean/std functions and names
		
		def radius(el):
			ox_state = metal_lookups.get(el.name)['oxidation_state']
			return el.data['Shannon radii'][f'{ox_state}']['VI'][''][self.radius_type]
		
		def cation_X(el):
			ox_state = metal_lookups.get(el.name)['oxidation_state']
			r = radius(el)
			return ox_state/r**2
		
		metal_funcs = {'oxide_Hf': lambda el: metal_lookups.get(el.name)['oxide_Hf'],
					   'MO_BondEnergy':lambda el: metal_lookups.get(el.name)['MO_BondEnergy'],
					   'MN_BondEnergy':lambda el: metal_lookups.get(el.name)['MN_BondEnergy'],
					   'MH_BondEnergy':lambda el: metal_lookups.get(el.name)['MH_BondEnergy'],
					   'ON_BondEnergyDelta': lambda el: metal_lookups.get(el.name)['ON_BondEnergyDelta'],
					   'OH_BondEnergyDelta': lambda el: metal_lookups.get(el.name)['OH_BondEnergyDelta'],
					   'NH_BondEnergyDelta': lambda el: metal_lookups.get(el.name)['NH_BondEnergyDelta'],
					   'MO_BondIonicity': lambda el: 1 - np.exp(-0.25*(el.X - mg.Element('O').X)**2),
					   'M_X': lambda el: el.X,
					   'M_CationX': cation_X,
					   'M_radius': radius,
					   'M_WorkFunction': lambda el: metal_lookups.get(el.name)['work_function'],
					   'M_sigma_elec': lambda el: elec_conductivity.get(el.name)
					  }
		
		for name,func in metal_funcs.items():
			mean = self.metal_mean_func(func)
			std = self.metal_std_func(func,mean=mean)
			features[f'{name}_mean'] = mean
			features[f'{name}_std'] = std
		
		# phase diagram feature - in BA-CaO-B3A triangle?
		# features['in_phase_triangle'] = self.is_in_phase_triangle()
		
		return features
		
	def feature_units(self):
		units = ['none',
				'energy',
				'energy',
				'energy',
				'energy',
				'energy',
				'energy',
				'energy',
				'energy',
				'energy',
				'energy',
				'energy',
				'energy',
				'none',
				'none',
				'none',
				'none',
				'none',
				'none',
				'length',
				'length',
				'energy',
				'energy',
				'S/cm',
				'S/cm',
				'energy',
				'none']
		return units
	
	def citations(self):
		cite = [
			# work function citations
			"@Inbook{Holzl1979,"
				"author={H{\"o}lzl, J. and Schulte, F. K.},"
				"editor={H{\"o}lzl, Josef and Schulte, Franz K. and Wagner, Heribert},"
				"title={Work function of metals},"
				"bookTitle={Solid Surface Physics},"
				"year={1979},"
				"publisher={Springer Berlin Heidelberg},"
				"address={Berlin, Heidelberg},"
				"pages={1--150},"
				"isbn={978-3-540-35253-2},"
				"doi={10.1007/BFb0048919},"
				"url={https://doi.org/10.1007/BFb0048919}"
			"}",
			"@Inbook{doi:10.1080/00222346908205102,"
				"author={Riviere, J.C.},"
				"editor = {Mino Green},"
				"title={Work Function: Measurement and Results}"
				"bookTitle = {Solid State Surface Science, Volume 1},"
				"year  = {1969},"
				"publisher = {Marcel Dekker},"
				"address={New York, NY, USA},"
				"pages={179},"
			"}",
			"@article{doi:10.1063/1.323539,"
				"author = {Michaelson,Herbert B. },"
				"title = {The work function of the elements and its periodicity},"
				"journal = {Journal of Applied Physics},"
				"volume = {48},"
				"number = {11},"
				"pages = {4729-4733},"
				"year = {1977},"
				"doi = {10.1063/1.323539},"
				"URL = {https://doi.org/10.1063/1.323539}"
			"}",
			# elec conductivity citation
			"@misc{AngstromSciences,"
				"author={Angstrom Sciences},"
				"title = {Elements Electrical Conductivity Reference Table},"
				"URL= {https://www.angstromsciences.com/elements-electrical-conductivity},"
				"note = {Accessed: 2019-02-27}"
			"}",
			# cation electronegativity citation
			"@article{Zohourian2017,"
				"author = {Zohourian, R. and Merkle, R. and Maier, J.},"
				"doi = {10.1016/j.ssi.2016.09.012},"
				"issn = {01672738},"
				"journal = {Solid State Ionics},"
				"pages = {64--69},"
				"publisher = {Elsevier B.V.},"
				"title = {{Proton uptake into the protonic cathode material BaCo0.4Fe0.4Zr0.2O3-$\delta$and comparison to protonic electrolyte materials}},"
				"url = {http://dx.doi.org/10.1016/j.ssi.2016.09.012},"
				"volume = {299},"
				"year = {2017}"
			"}",
			# BCA phase diagram citation
			# "@article{Zhang2017,"
				# "author = {Zhang, Rui and Mao, Huahai and Taskinen, Pekka},"
				# "doi = {10.1111/jace.14793},"
				# "issn = {15512916},"
				# "journal = {Journal of the American Ceramic Society},"
				# "keywords = {glass-ceramics,phase equilibria,thermodynamics},"
				# "number = {6},"
				# "pages = {2722--2731},"
				# "title = {{Phase equilibria study and thermodynamic description of the BaO-CaO-Al2O3 system}},"
				# "volume = {100},"
				# "year = {2017}"
			# "}"
			# pymatgen citation
			"@article{Ong2012b,"
				"author = {Ong, Shyue Ping and Richards, William Davidson and Jain, Anubhav and Hautier, Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and Chevrier, Vincent L. and Persson, Kristin A. and Ceder, Gerbrand},"
				"doi = {10.1016/j.commatsci.2012.10.028},"
				"file = {:Users/shyue/Mendeley Desktop/Ong et al/Computational Materials Science/2013 - Ong et al. - Python Materials Genomics (pymatgen) A robust, open-source python library for materials analysis.pdf:pdf;:Users/shyue/Mendeley Desktop/Ong et al/Computational Materials Science/2013 - Ong et al. - Python Materials Genomics (pymatgen) A robust, open-source python library for materials analysis(2).pdf:pdf},"
				"issn = {09270256},"
				"journal = {Computational Materials Science},"
				"month = feb,"
				"pages = {314--319},"
				"title = {{Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis}},"
				"url = {http://linkinghub.elsevier.com/retrieve/pii/S0927025612006295},"
				"volume = {68},"
				"year = {2013}"
			"}"
			]
		return list(np.unique(cite + mpcalc.citations()))
				
				
		
class BCA_Featurizer(BaseFeaturizer):
	def __init__(self,radius_type='ionic_radius',normalize_formula=False):
		self.radius_type = radius_type
		self.normalize_formula = normalize_formula
		self.ValenceOrbital = ValenceOrbital()
		self.AtomicOrbitals = AtomicOrbitalsMod()
		self.CohesiveEnergy = CohesiveEnergy()
		self.BandCenter = BandCenter()
		self.ValenceOrbitalEnergy = ValenceOrbitalEnergy()
		#custom ElementProperty featurizer
		elemental_properties = ['BoilingT', 'MeltingT',
			'BulkModulus', 'ShearModulus', 
			'Row', 'Column', 'Number', 'MendeleevNumber', 'SpaceGroupNumber',
			'Density','MolarVolume',
			'FusionEnthalpy','HeatVaporization',
			'Polarizability', 
			'ThermalConductivity']
		self.ElementProperty = ElementProperty(data_source='magpie',features=elemental_properties,
						  stats=["mean", "std_dev"])
		#check matminer featurizers
		self.check_matminer_featurizers()
		
	def featurize(self,composition):
		bca = BCA(composition,self.radius_type,self.normalize_formula)
		bca_features = bca.featurize()
		
		vo_features = self.ValenceOrbital.featurize(bca.metal_composition) #avg and frac s, p , d, f electrons for metals
		vo_features += [sum(vo_features[0:3])] #avg total valence electrons for metals
		ao_features = self.AtomicOrbitals.featurize(bca.metal_composition) #HOMO and LUMO character and energy levels for metals from atomic orbitals)
		ao_features = [ao_features[i] for i in range(len(ao_features)) if i not in (0,1,3,4)]#exclude HOMO_character,HOMO_element, LUMO_character, LUMO_element - categoricals
		ce_features = self.CohesiveEnergy.featurize(bca.metal_composition,formation_energy_per_atom=1e-10) #avg metal elemental cohesive energy
		bc_features = self.BandCenter.featurize(bca.metal_composition) + self.BandCenter.featurize(bca.composition)
		ve_features = self.ValenceOrbitalEnergy.featurize(bca.metal_composition) + self.ValenceOrbitalEnergy.featurize(bca.composition)
		ep_features = self.ElementProperty.featurize(bca.metal_composition) + self.ElementProperty.featurize(bca.composition)
		
		mm_features = vo_features + ao_features + ce_features + bc_features + ve_features + ep_features 
		
		return list(bca_features.values()) + mm_features
		
	@property
	def ElementProperty_custom_labels(self):
		"""
		Generate custom labels for ElementProperty featurizer that follow same naming convention as Perovskite class
		"""
		elemental_property_label_map = {'BoilingT':'boil_temp','MeltingT':'melt_temp',
							'BulkModulus':'bulk_mod','ShearModulus':'shear_mod',
							'Row':'row','Column':'column','Number':'number','MendeleevNumber':'mendeleev','SpaceGroupNumber':'space_group',
							'Density':'density','MolarVolume':'molar_vol',
							'FusionEnthalpy':'H_fus','HeatVaporization':'H_vap',
							'Polarizability':'polarizability',
							'ThermalConductivity':'sigma_therm'}
							
		element_property_labels = list(map(elemental_property_label_map.get,self.ElementProperty.features))
		labels = []
		for attr in element_property_labels:
			for stat in self.ElementProperty.stats:
				if stat=='std_dev':
					stat = 'std'
				labels.append(f'M_{attr}_{stat}')
		for attr in element_property_labels:
			for stat in self.ElementProperty.stats:
				if stat=='std_dev':
					stat = 'std'
				labels.append(f'BCA_{attr}_{stat}')
		return labels
		
	@property
	def ElementProperty_units(self):
		"""
		Generate units for ElementProperty featurizer that follow same naming convention as Perovskite class
		"""
		elemental_property_unit_map = {'BoilingT':'temperature','MeltingT':'temperature',
							'BulkModulus':'pressure','ShearModulus':'pressure',
							'Row':'none','Column':'none','Number':'none','MendeleevNumber':'none','SpaceGroupNumber':'none',
							'Density':'density','MolarVolume':'volume',
							'FusionEnthalpy':'energy','HeatVaporization':'energy',
							'Polarizability':'polarizability',
							'ThermalConductivity':'therm'}
							
		element_property_units = list(map(elemental_property_unit_map.get,self.ElementProperty.features))
		units = []
		for ep_unit in element_property_units:
			for stat in self.ElementProperty.stats:
				units.append(ep_unit)
		return units*2
		
	def ElementProperty_label_check(self):
		"""
		Check that ElementProperty feature labels are as expected
		If not, features may not align with feature labels
		"""
		#ElementProperty.feature_labels() code as of 1/24/20
		labels = []
		for attr in self.ElementProperty.features:
			src = self.ElementProperty.data_source.__class__.__name__
			for stat in self.ElementProperty.stats:
				labels.append("{} {} {}".format(src, stat, attr))
		
		if labels!=self.ElementProperty.feature_labels():
			raise Exception('ElementProperty features or labels have changed')
	
	
	@property
	def matminer_labels(self):
		"""
		Feature labels for matminer-derived features
		"""
		labels = [
			#ValenceOrbital labels
			'M_ValenceElec_s_mean',
			'M_ValenceElec_p_mean',
			'M_ValenceElec_d_mean',
			'M_ValenceElec_f_mean',
			'M_ValenceElec_s_frac',
			'M_ValenceElec_p_frac',
			'M_ValenceElec_d_frac',
			'M_ValenceElec_f_frac',
			'M_ValenceElec_tot_mean',
			#AtomicOrbitals labels
			#'M_HOMO_character',
			'M_HOMO_energy',
			#'M_LUMO_character',
			'M_LUMO_energy',
			'M_AO_gap',
			#CohesiveEnergy labels
			'M_cohesive_energy_mean',
			#BandCenter labels
			'M_BandCenter',
			'BCA_BandCenter',
			#ValenceOrbitalEnergy labels
			'M_ValenceEnergy_mean',
			'BCA_ValenceEnergy_mean'
			]
			
		labels += self.ElementProperty_custom_labels
		
		return labels	
	
	@property
	def matminer_units(self):
		"""
		Feature units for matminer-derived features
		"""
		units = [
			#ValenceOrbital units
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			#AtomicOrbitals units
			#'M_HOMO_character',
			'energy',
			#'M_LUMO_character',
			'energy',
			'energy',
			#CohesiveEnergy units
			'energy',
			#BandCenter units
			'energy',
			'energy',
			#ValenceOrbitalEnergy units
			'energy',
			'energy'
			]
			
		units += self.ElementProperty_units
		
		return units
	
	def feature_labels(self):
		bca_feature_labels = list(BCA(mg.Composition('BaO'),self.radius_type,self.normalize_formula).featurize().keys())
		
		return bca_feature_labels + self.matminer_labels
		
	def feature_units(self):
		bca_units = BCA(mg.Composition('BaO')).feature_units()
		
		return bca_units + self.matminer_units
		
	def check_matminer_featurizers(self):
		"""
		Check that features and feature order for matminer featurizers are as expected
		If features or feature order have changed, featurize() may return unexpected features that do not align with feature_labels()
		"""
		#verify that matminer feature labels haven't changed
		if self.ValenceOrbital.feature_labels() != ['avg s valence electrons',
											 'avg p valence electrons',
											 'avg d valence electrons',
											 'avg f valence electrons',
											 'frac s valence electrons',
											 'frac p valence electrons',
											 'frac d valence electrons',
											 'frac f valence electrons']:
			raise Exception('ValenceOrbital features or labels have changed')
			
		if self.AtomicOrbitals.feature_labels() != ['HOMO_character',
											 'HOMO_element',
											 'HOMO_energy',
											 'LUMO_character',
											 'LUMO_element',
											 'LUMO_energy',
											 'gap_AO']:
			raise Exception('AtomicOrbitals features or labels have changed')

		if self.CohesiveEnergy.feature_labels() != ['cohesive energy']:
			raise Exception('CohesiveEnergy features or labels have changed')
			
		if self.BandCenter.feature_labels() != ['band center']:
			raise Exception('BandCenter features or labels have changed')
	
		self.ElementProperty_label_check()
		
	def citations(self):
		featurizers = [self.ValenceOrbital, self.AtomicOrbitals, self.CohesiveEnergy, self.BandCenter, self.ValenceOrbitalEnergy, BCA(mg.Composition('BaO'))]
		return list(np.unique(sum([f.citations() for f in featurizers],[])))
		
class GenericFeaturizer(BaseFeaturizer):
	"""
	Featurizer to use generic properties available in matminer featurizers; no features from BCA class utilized
	"""
	def __init__(self,normalize_formula=False):
		self.normalize_formula = normalize_formula
		# don't need ValenceOrbital - valence counts etc. covered in ElementProperty.from_preset('magpie')
		# self.ValenceOrbital = ValenceOrbital()
		self.AtomicOrbitals = AtomicOrbitalsMod()
		self.CohesiveEnergy = CohesiveEnergy()
		self.BandCenter = BandCenter()
		self.ValenceOrbitalEnergy = ValenceOrbitalEnergy()
		# ElementProperty featurizer with magpie properties plus additional properties
		self.ElementProperty = ElementProperty.from_preset('magpie')
		self.ElementProperty.features += ['BoilingT', 
					'BulkModulus', 'ShearModulus', 
					'Density','MolarVolume',
					'FusionEnthalpy','HeatVaporization',
					'Polarizability', 
					'ThermalConductivity']
		# range, min, max are irrelevant inside the ternary
		# self.ElementProperty.stats = ['mean', 'avg_dev','mode']

		# check matminer featurizers
		self.check_matminer_featurizers()
		
	def featurize(self,composition):
		# use BCA just to get composition and metal_composition
		bca = BCA(composition,'ionic_radius',self.normalize_formula)
		
		ao_features = self.AtomicOrbitals.featurize(bca.metal_composition) # HOMO and LUMO character and energy levels for metals from atomic orbitals)
		ao_features = [ao_features[i] for i in range(len(ao_features)) if i not in (0,1,3,4)] # exclude HOMO_character,HOMO_element, LUMO_character, LUMO_element - categoricals
		ce_features = self.CohesiveEnergy.featurize(bca.metal_composition,formation_energy_per_atom=1e-10) # avg metal elemental cohesive energy
		bc_features = self.BandCenter.featurize(bca.metal_composition) + self.BandCenter.featurize(bca.composition)
		ve_features = self.ValenceOrbitalEnergy.featurize(bca.metal_composition) + self.ValenceOrbitalEnergy.featurize(bca.composition)
		ep_features = self.ElementProperty.featurize(bca.metal_composition) + self.ElementProperty.featurize(bca.composition)
		
		mm_features = ao_features + ce_features + bc_features + ve_features + ep_features 
		
		return mm_features
	
	def feature_labels(self):
		"""
		Feature labels for matminer-derived features
		"""
		labels = [
			#AtomicOrbitals labels
			#'M_HOMO_character',
			'M_HOMO_energy',
			#'M_LUMO_character',
			'M_LUMO_energy',
			'M_AO_gap',
			#CohesiveEnergy labels
			'M_cohesive_energy_mean',
			#BandCenter labels
			'M_BandCenter',
			'BCA_BandCenter',
			#ValenceOrbitalEnergy labels
			'M_ValenceEnergy_mean',
			'BCA_ValenceEnergy_mean'
			]
			
		labels += [f'M {l}' for l in self.ElementProperty.feature_labels()]
		labels += [f'BCA {l}' for l in self.ElementProperty.feature_labels()]
		
		return labels	
	
	@property
	def matminer_units(self):
		"""
		Feature units for matminer-derived features
		"""
		units = [
			#ValenceOrbital units
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			#AtomicOrbitals units
			#'M_HOMO_character',
			'energy',
			#'M_LUMO_character',
			'energy',
			'energy',
			#CohesiveEnergy units
			'energy',
			#BandCenter units
			'energy',
			'energy',
			#ValenceOrbitalEnergy units
			'energy',
			'energy'
			]
			
		units += self.ElementProperty_units
		
		return units
		
	def feature_units(self):
		bca_units = BCA(mg.Composition('BaO')).feature_units()
		
		return bca_units + self.matminer_units
		
	def check_matminer_featurizers(self):
		"""
		Check that features and feature order for matminer featurizers are as expected
		If features or feature order have changed, featurize() may return unexpected features that do not align with feature_labels()
		"""
		#verify that matminer feature labels haven't changed
		if self.AtomicOrbitals.feature_labels() != ['HOMO_character',
											 'HOMO_element',
											 'HOMO_energy',
											 'LUMO_character',
											 'LUMO_element',
											 'LUMO_energy',
											 'gap_AO']:
			raise Exception('AtomicOrbitals features or labels have changed')

		if self.CohesiveEnergy.feature_labels() != ['cohesive energy']:
			raise Exception('CohesiveEnergy features or labels have changed')
			
		if self.BandCenter.feature_labels() != ['band center']:
			raise Exception('BandCenter features or labels have changed')
			
	def citations(self):
		featurizers = [self.AtomicOrbitals, self.CohesiveEnergy, self.BandCenter, self.ValenceOrbitalEnergy]
		citations = sum([f.citations() for f in featurizers],[])
		# add pymatgen citation
		citations += [
			"@article{Ong2012b,"
				"author = {Ong, Shyue Ping and Richards, William Davidson and Jain, Anubhav and Hautier, Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and Chevrier, Vincent L. and Persson, Kristin A. and Ceder, Gerbrand},"
				"doi = {10.1016/j.commatsci.2012.10.028},"
				"file = {:Users/shyue/Mendeley Desktop/Ong et al/Computational Materials Science/2013 - Ong et al. - Python Materials Genomics (pymatgen) A robust, open-source python library for materials analysis.pdf:pdf;:Users/shyue/Mendeley Desktop/Ong et al/Computational Materials Science/2013 - Ong et al. - Python Materials Genomics (pymatgen) A robust, open-source python library for materials analysis(2).pdf:pdf},"
				"issn = {09270256},"
				"journal = {Computational Materials Science},"
				"month = feb,"
				"pages = {314--319},"
				"title = {{Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis}},"
				"url = {http://linkinghub.elsevier.com/retrieve/pii/S0927025612006295},"
				"volume = {68},"
				"year = {2013}"
			"}"
			]
		return list(np.unique(citations))