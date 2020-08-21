import pandas as pd
import numpy as np
import pymatgen as mg
import matplotlib.pyplot as plt
import os
import re

# ----------
# Data load
# ----------
def load_data(file):
	"Load data from file"
	df = pd.read_csv(file,skiprows=19)
	#remove pipe separator columns
	pipecols = [c for c in df.columns if c[:2]==' |']
	df = df.drop(pipecols,axis=1)
	#trim column names
	df.columns = [c.strip() for c in df.columns]
	df['tot-flow'] = df['H2-flow'] + df['N2-flow']
	# equilibrium conversion rate at T
	df['eq_ppm'] = df['Outer-Temp'].map(get_eq_conv)
	# fraction of equilibrium conversion achieved
	df['rel_conv'] = df['NH3_ppm']/df['eq_ppm']
	# relative rate: prod_rate/eq_ppm is proportional to (NH3_ppm/eq_ppm)*tot-flow
	df['rel_rate'] = df['NH3_Prod_rate']/df['eq_ppm']
	return df

def load_metadata(file):
	"Load file metadata"
	with open(file,'r') as f:
		txt = f.read()
	metadata = {}

	#configuration
	config_start = txt.find('\n') + 1
	config_end = txt.find('# Catalyst:') - 1
	config = txt[config_start:config_end]
	metadata['Config'] = {k:v for k,v in [cfl.replace('#		  ','').split(':  ') for cfl in config.split('\n')]}

	def field_text(txt,start_string,end_string):
		start = txt.find(start_string) + len(start_string)
		end = txt[start:].find(end_string) + start
		return txt[start:end]

	metadata['Catalyst Description'] = field_text(txt,'Catalyst_Description: ','\n')
	metadata['Catalyst Mass'] = float(field_text(txt,'Catalyst mass: ','\n'))
	metadata['Notes'] = field_text(txt,'Notes: ','\n')

	return metadata
	
def get_file_list(datadir,config_file='non_BCA_files.xlsx'):
	"""Get list of usable data files according to config file"""
	# load config file to determine which files to load besides standard BCAxyz formatted files
	config = pd.read_excel(os.path.join(datadir,config_file))
	config = config[config['ignore']!=1]
	add_files = dict(zip(config['filename'],config['formula']))
	# get list of files to load
	filter_func = lambda x: (x[:3]=='BCA' and x[-3:] in ('txt','csv')) or x in add_files.keys()
	files = [f for f in os.listdir(datadir) if filter_func(f)]
	return files 
	
def load_dir(datadir, aggregate_cols=['NH3_Prod_rate','NH3_ppm'],aggregate_kwargs=[{'window':301,'aggregate':'max'},{'window':301,'aggregate':'max'}], 
			 config_file='non_BCA_files.xlsx'):
	# load config file to determine which files to load besides standard BCAxyz formatted files
	config = pd.read_excel(os.path.join(datadir,config_file))
	config = config[config['ignore']!=1]
	add_files = dict(zip(config['filename'],config['formula']))
	# get list of files to load
	filter_func = lambda x: (x[:3]=='BCA' and x[-3:] in ('txt','csv')) or x in add_files.keys()
	files = [f for f in os.listdir(datadir) if filter_func(f)]
	
	data = pd.DataFrame(files,columns=['filename'])
	
	# get composition
	data['composition'] = [get_file_comp(f,add_files) for f in files]
	# add formula for DataFrame readability
	data['formula'] = data['composition'].map(lambda x: x.formula)
	# add BCA str
	data['BCA_str'] = data['composition'].map(lambda x: BCA_str_from_comp(x))
	
	# aggregate specified columns
	for agg_col,agg_kw in zip(aggregate_cols,aggregate_kwargs):
		file_paths = [os.path.join(datadir,file) for file in files]
		if agg_col=='NH3_Prod_rate' and agg_kw['aggregate']=='max':
			vals = np.zeros(len(file_paths))
			temps = vals.copy()
			HN_ratio = vals.copy()
			flows = vals.copy()
			for j,file in enumerate(file_paths):
				df = load_data(file)
				wdf = windowed_median(df,agg_col,agg_kw['window'])
				vals[j] = wdf[agg_col].max()
				# get T corresponding to max prod rate
				temps[j] = wdf.loc[wdf[agg_col].idxmax(),'Outer-Temp']
				# get H2:N2 ratio at max prod rate
				HN_ratio[j] = wdf.loc[wdf[agg_col].idxmax(),'H2-flow']/wdf.loc[wdf[agg_col].idxmax(),'N2-flow']
				# get total flow at max prod rate
				flows[j] = wdf.loc[wdf[agg_col].idxmax(),'tot-flow']
			data['{}_{}'.format(agg_kw.get('aggregate'),agg_col)] = vals
			data['T_max'] = temps
			data['HN_ratio_max'] = HN_ratio
			data['flow_max'] = flows
		else:
			vals = aggregate_windowed_median(file_paths,agg_col,**agg_kw)
			data['{}_{}'.format(agg_kw.get('aggregate'),agg_col)] = vals
		
	return data
	
# ---------------
# File info
# ---------------	
def BCA_formula_from_str(BCA_str):
	"""
	Get chemical formula string from BCA string
	
	Args:
		BCA_str: BCA ratio string (e.g. 'BCA311')
	"""
	if len(BCA_str)==6 and BCA_str[:3]=='BCA':
		# format: BCAxyz. suitable for single-digit integer x,y,z
		funits = BCA_str[-3:]
	else:
		# format: BxCyAz. suitable for multi-digit or non-integer x,y,z
		funits = re.split('[BCA]',BCA_str)
		funits = [u for u in funits if len(u) > 0]
		funits
	components = ['BaO','CaO','Al2O3']
	formula = ''.join([f'({c}){n}' for c,n in zip(components, funits)])
	return formula
	
def BCA_str_from_comp(comp):
	"""
	Get BCA string from composition
	
	Args:
		comp: pymatgen composition
	"""
	base_amt = {'Ba':1,'Ca':1,'Al':2}
	amts = np.array([comp[k]/v for k,v in base_amt.items()]).astype(int)
	# check oxygen amount matches metal amounts
	O_units = [1,1,3]
	if np.dot(O_units,amts)!= comp['O']:
		raise Exception('Non-BCA composition. Oxygen stoichiometry is off')
	
	# reduce amounts
	div = np.gcd.reduce(amts) # greatest common divisor
	amts = (amts/div).astype(int)
	return 'B{}C{}A{}'.format(*amts)
	
def pretty_string_from_comp(comp):
	base_amt = {'Ba':1,'Ca':1,'Al':2}
	amts = np.array([comp[k]/v for k,v in base_amt.items()]).astype(int)
	# check oxygen amount matches metal amounts
	O_units = [1,1,3]
	if np.dot(O_units,amts)!= comp['O']:
		raise Exception('Non-BCA composition. Oxygen stoichiometry is off')
	
	# reduce amounts
	div = np.gcd.reduce(amts) # greatest common divisor
	amts = (amts/div).astype(int)
	return 'B$_{{{}}}$C$_{{{}}}$A$_{{{}}}$'.format(*amts)
	
def get_file_comp(file,nonBCA_dict={}):
	"""
	Construct pymatgen composition from filename
	
	Args:
		file: filename
		nonBCA_dict: dict of formulas for files without standard BCA strings
	"""
	try:
		comp = mg.Composition(nonBCA_dict[file])
	except KeyError:
		BCA_str = file[:6]
		comp = mg.Composition(BCA_formula_from_str(BCA_str))
		
	return comp
	
def disp_func(file):
	"""
	Get first part of filename (usually BCA string) for display
	"""
	disp = file[:file.find('_')]
	end = disp.find('.')
	if end > 0:
		disp = disp[:end]
	return disp	
	
# ----------------
# Equilibrium calc
# ----------------
T_conv = np.concatenate(([300],np.arange(400,601,50))) # T in C
K_eq = np.array([4.34e-3,1.64e-4,4.51e-5,1.45e-5,5.38e-6,2.25e-6]) # equil. constants for N2 + 3H2 --> 2NH3
p_h2 = 7.5 # init. H2 pressure (10 bar total)
p_n2 = 2.5 # init. N2 pressure

# solve for equilibrium P_NH3
C0 = K_eq*p_h2**3*p_n2
C1 = -K_eq*(p_h2**3 + 9*p_h2**2*p_n2)
C2 = K_eq*(9*p_h2**2 + 27*p_h2*p_n2) - 4
C3 = -K_eq*(27*p_h2 + 27*p_n2)
C4 = 27*K_eq

coefs = [[c4,c3,c2,c1,c0] for c4,c3,c2,c1,c0 in zip(C4,C3,C2,C1,C0)]
delta_p = np.array([np.roots(c)[3].real for c in coefs])
p_nh3 = 2*delta_p # eq. NH3 pressure
eq_conv = 1e6*p_nh3/(p_h2+p_n2-2*delta_p) # eq. conversion in ppm

def get_eq_conv(T):
	return np.interp(T,T_conv,eq_conv)

# ---------------
# Data processing
# ---------------
def windowed_median(df,col,window):
	"""
	Split data into windows and return median of col (and other data from corresponding row) for each window
	
	Args:
		df: DataFrame 
		col: data column to use for median
		window: number of points per window for median. Should be odd
	"""
	wdf = df.copy()
	wdf['window'] = (wdf.index/window).astype(int)
	jdf = wdf.join(wdf.groupby('window').median()[col],on='window',rsuffix='_window')
	jdf = jdf[jdf[col]==jdf[col+'_window']]
	return jdf
	
def aggregate_windowed_median(files,col,window,aggregate,show_plot=True,sharex=True):
	agg_col = []

	if show_plot==True:
		ncol = 3
		nrow = int(np.ceil(len(files)/ncol))
		fig, axes = plt.subplots(nrow,ncol,figsize=(ncol*2.5,nrow*2+1),sharex=sharex,sharey=True)
		def disp_func(file):
			fname = os.path.basename(file)
			disp = fname[:fname.find('_')]
			end = disp.find('.')
			if end > 0:
				disp = disp[:end]
			return disp
	
	for i,file in enumerate(files):
		df = load_data(file)
		mdf = windowed_median(df,col,window)
		
		if show_plot==True:
			ax = axes.ravel()[i]
			p1 = ax.plot(df['Elapsed time']/3600,df[col],'.',label='Raw',markersize=1)
			p2 = ax.plot(mdf['Elapsed time']/3600,mdf[col],label='Windowed Median')
			ax.set_title(disp_func(file))
		agg_c = getattr(mdf[col],aggregate)()
		ax.text(0.1,0.9,'{} value: {}'.format(aggregate,round(agg_c,1)),transform=ax.transAxes)
		agg_col.append(agg_c)

	if show_plot==True:
		for ax in axes[:,0]:
			ax.set_ylabel(col)

		for ax in axes[nrow-1,:]:
			ax.set_xlabel('Time (h)')

		for ax in axes.ravel()[len(files):]:
			ax.axis('off')

		fig.tight_layout()
		fig.subplots_adjust(bottom=1/(nrow*2.5 + 1))
		fig.legend(labels=('Raw','Windowed median'),handles=(p1[0],p2[0]),loc='lower center',ncol=2)
		
	return agg_col