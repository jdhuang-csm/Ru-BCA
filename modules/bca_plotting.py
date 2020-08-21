import numpy as np
import ternary
from ternary.helpers import simplex_iterator
import matplotlib.pyplot as plt
import pymatgen as mg
import pandas as pd
from file_load import BCA_formula_from_str
from model_eval import predict_interval

def get_coords_from_comp(comp,tern_axes=['Ca','Al','Ba']):
	base_amt = {'Ba':1,'Ca':1,'Al':2}
	coords = np.array([comp[el]/base_amt[el] for el in tern_axes])
	coords = coords/np.sum(coords)
	return coords
	
def get_comp_from_coords(coords,tern_axes=['Ca','Al','Ba'],scale=1):
	if len(coords)==2:
		a,b = coords
		c = scale - a- b
		coords = (a,b,c)
	# else:
		# a,b,c = coords
		
	oxides = {'Ba':'BaO','Ca':'CaO','Al':'Al2O3','B':'B2O3','Mg':'MgO','Sr':'SrO'}
	formula = ''.join(['({}){}'.format(oxides[m],amt) for m,amt in zip(tern_axes,coords)])
	return mg.Composition(formula)
	
def add_colorbar(fig=None, cbrect=[0.9,0.15,0.02,0.75], label=None, tickformat=None, 
				 cmap = None, vlim=None, norm=None,
				 tick_params={}, label_kwargs={},subplots_adjust={'left':0.05,'wspace':0.35, 'hspace':0.25, 'right':0.8},
				 **cb_kwargs):
	#add a single colorbar
	if fig is None:
		fig = plt.gcf()
	#make an axis for colorbar to control position/size
	cbaxes = fig.add_axes(cbrect) #[left, bottom, width, height]
	#code from ternary.colormapping.colorbar_hack
	
	if vlim is not None:
		vmin,vmax = vlim
	else:
		vmin = None
		vmax = None
	
	if norm==None:
		# if logscale==True:
			# norm = colors.LogNorm(vmin=vmin,vmax=vmax)
		norm = plt.Normalize(vmin=vmin,vmax=vmax)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm._A = []
	cb = fig.colorbar(sm, cax=cbaxes, format=tickformat,**cb_kwargs)
	cb.ax.tick_params(**tick_params)
	if label is not None:
		cb.set_label(label, **label_kwargs)

	fig.subplots_adjust(**subplots_adjust)
	
def plot_labeled_ternary(comps,values,ax=None,label_points=True,add_labeloffset=0,corner_labelsize=12,point_labelsize=11,point_labeloffset=[0,0.01,0],cmap=None,vlim=None,**scatter_kw):
	tern_axes = ['Ca','Al','Ba']
	
	if ax is None:
		fig, ax = plt.subplots(figsize=(9,8))
	else:
		fig = ax.get_figure()
		
	#tfig, tax = ternary.figure(scale=1,ax=ax)
	tax = ternary.TernaryAxesSubplot(scale=1,ax=ax)

	points = [get_coords_from_comp(c,tern_axes) for c in comps]
	
	if vlim is None:
		vmin=min(values)
		vmax=max(values)
	else:
		vmin, vmax = vlim
	tax.scatter(points,c=values,cmap=cmap,vmin=vmin,vmax=vmax,**scatter_kw)

	tern_labels = ['CaO','Al$_2$O$_3$','BaO']
	
	tax.right_corner_label(tern_labels[0],fontsize=corner_labelsize,va='center',offset=0.08+add_labeloffset)
	tax.top_corner_label(tern_labels[1],fontsize=corner_labelsize,va='center',offset=0.05+add_labeloffset)
	tax.left_corner_label(tern_labels[2],fontsize=corner_labelsize,va='center',offset=0.08+add_labeloffset)

	tax.boundary(linewidth=1)
	#tax.clear_matplotlib_ticks()
	ax.axis('off')

	if label_points==True:
		for p,val in zip(points,values):
			if pd.isnull(val):
				disp = 'NA'
			else:
				disp = '{}'.format(int(round(val,0)))
			tax.annotate(disp,p+point_labeloffset,size=point_labelsize,ha='center',va='bottom')

	#add_colorbar(fig,label='NH3 Production Rate (mmol/g$\cdot$h)',vmin=min(values),vmax=max(values),cbrect=[0.9,0.2,0.03,0.67])
	tax._redraw_labels()
	
	return tax

	
def featurize_simplex(scale, featurizer, feature_cols=None, scaler=None,tern_axes=['Ca','Al','Ba']):
	"""
	Generate feature matrix for simplex (intended for making heatmaps)
	
	Args:
		scale: simplex scale. Determines point density
		featurizer: matminer-like featurizer instance
		feature_cols: subset of column names to use for features. If None, use all columns
		scaler: fitted scaler instance. If None, feature matrix will not be scaled
		tern_axes: ternary axes. Default ['Ca','Al','Ba']
	Returns:
		coords: list of simplex coordinates
		X: feature matrix
	"""
	
	coords = [tup for tup in simplex_iterator(scale)]
	comps = [get_comp_from_coords(c,tern_axes=tern_axes) for c in coords]
	df = pd.DataFrame([[comp] for comp in comps],columns=['composition'])
	featurizer.featurize_dataframe(df,col_id='composition',inplace=True)
	if feature_cols is None:
		X = df
	else:
		X = df.loc[:,feature_cols]
	
	if scaler is not None:
		X = pd.DataFrame(scaler.transform(X),columns=X.columns)
		
	return coords, X

def predict_simplex(estimator, scale, featurizer=None, feature_cols=None, scaler=None,use_X=None,tern_axes=['Ca','Al','Ba'],metric='median'):
	"""
	Generate predictions for simplex (intended for making heatmaps)
	
	Args:
		estimator: fitted estimator
		scale: simplex scale. Determines point density
		featurizer: matminer-like featurizer instance
		feature_cols: subset of column names to use for features. If None, use all columns
		scaler: fitted scaler instance. If None, feature matrix will not be scaled
		use_X: optional arg to provide feature matrix if already calculated. 
			If provided, featurizer, feature_cols, and scaler will be ignored
		tern_axes: ternary axes. Default ['Ca','Al','Ba']
		metric: if 'median', return point estimate. If 'iqr', return IQR of prediction
	Returns:
		coords: list of simplex coordinates
		y: estimator predictions
	"""
	if use_X is None:
		coords, X = featurize_simplex(scale,featurizer,feature_cols,scaler,tern_axes)
	else:
		coords = [tup for tup in simplex_iterator(scale)]
		X = use_X
		
	if type(X) == pd.core.frame.DataFrame:
		X = X.values
		
	# handle nans and infs
	# find rows with any nan or inf values
	bad_val_idx = np.max(np.array([np.max(np.isinf(X),axis=1),np.max(np.isnan(X),axis=1)]),axis=0)
	if np.sum(bad_val_idx) > 0:
		print('Warning: feature matrix contains nans or infs. Number of bad rows: {}'.format(np.sum(bad_val_idx)))
	# set all features in bad rows to zero so that they don't break estimator.predict()
	X[bad_val_idx] = 0
	
	if metric=='median':
		y = estimator.predict(X)
	elif metric=='iqr':
		lb,ub = predict_interval(estimator,X,0.682)
		y = ub - lb
	# set predictions for bad feature values to nan
	y[bad_val_idx] = np.nan
	
	return coords, y
	
def estimator_ternary_heatmap(scale, estimator, featurizer=None, feature_cols=None, scaler=None,use_X=None, style='triangular', 
					   labelsize=11, add_labeloffset=0, cmap=None, ax=None,figsize=None, vlim=None,metric='median',
					   multiple=0.1, tick_kwargs={'tick_formats':'%.1f','offset':0.02},
					   tern_axes=['Ca','Al','Ba'],tern_labels = ['CaO','Al$_2$O$_3$','BaO']):
	"""
	Generate ternary heatmap of ML predictions
	
	Args:
		scale: simplex scale
		estimator: sklearn estimator instance
		featurizer: featurizer instance
		feature_cols: subset of feature names used in model_eval
		scaler: sklearn scaler instance
		use_X: pre-calculated feature matrix; if passed, featurizer, feature_cols, and scaler are ignored
		style: heatmap interpolation style
		tern_axes: ternary axes. Only used for generating simplex compositions; ignored if use_X is supplied. Defaults to ['Ca','Al','Ba']
		metric: if 'median', return point estimate. If 'iqr', return IQR of prediction
	""" 
	coords, y = predict_simplex(estimator, scale, featurizer, feature_cols, scaler,use_X,tern_axes,metric)
	
	if vlim is None:
		vmin = min(y)
		vmax = max(y)
	else:
		vmin,vmax = vlim

	points = dict(zip([c[0:2] for c in coords],y))

	if ax==None:
		fig, ax = plt.subplots(figsize=figsize)
		tfig, tax = ternary.figure(scale=scale,ax=ax)
	else:
		tax = ternary.TernaryAxesSubplot(scale=scale,ax=ax)

	tax.heatmap(points,style=style,colorbar=False,cmap=cmap,vmin=vmin,vmax=vmax)
	#rescale_ticks(tax,new_scale=axis_scale,multiple = multiple, **tick_kwargs)
	tax.boundary()
	tax.ax.axis('off')

	tax.right_corner_label(tern_labels[0],fontsize=labelsize,va='center',offset=0.08+add_labeloffset)
	tax.top_corner_label(tern_labels[1],fontsize=labelsize,va='center',offset=0.05+add_labeloffset)
	tax.left_corner_label(tern_labels[2],fontsize=labelsize,va='center',offset=0.08+add_labeloffset)

	tax._redraw_labels()

	return tax
	
def feature_ternary_heatmap(scale, feature_name, featurizer=None, use_X=None, style='triangular', 
                       labelsize=11, add_labeloffset=0, cmap=None, ax=None,figsize=None, vlim=None,
                       multiple=0.1, tick_kwargs={'tick_formats':'%.1f','offset':0.02},
                       tern_axes=['Ca','Al','Ba'],tern_labels = ['CaO','Al$_2$O$_3$','BaO']):
    """
    
    """ 
    if use_X is None:
        coords, X = featurize_simplex(scale,featurizer,feature_cols=featurizer.feature_labels(),tern_axes=tern_axes)
        X = pd.DataFrame(X,columns=featurizer.feature_labels())
    else:
        coords = [tup for tup in simplex_iterator(scale)]
        X = use_X
    
    y = X.loc[:,feature_name]
    
    if vlim is None:
        vmin = min(y)
        vmax = max(y)
    else:
        vmin,vmax = vlim

    points = dict(zip([c[0:2] for c in coords],y))

    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
        tfig, tax = ternary.figure(scale=scale,ax=ax)
    else:
        tax = ternary.TernaryAxesSubplot(scale=scale,ax=ax)

    tax.heatmap(points,style=style,colorbar=False,cmap=cmap,vmin=vmin,vmax=vmax)
    #rescale_ticks(tax,new_scale=axis_scale,multiple = multiple, **tick_kwargs)
    tax.boundary()
    tax.ax.axis('off')

    tax.right_corner_label(tern_labels[0],fontsize=labelsize,va='center',offset=0.08+add_labeloffset)
    tax.top_corner_label(tern_labels[1],fontsize=labelsize,va='center',offset=0.05+add_labeloffset)
    tax.left_corner_label(tern_labels[2],fontsize=labelsize,va='center',offset=0.08+add_labeloffset)

    tax._redraw_labels()

    return tax
	
def ternary_scatter_vs_heatmap(scatter_comps,scatter_values, hmap_scale,hmap_estimator,vlim,cmap=None,
							   hmap_featurizer=None,hmap_feature_cols=None,hmap_scaler=None,hmap_use_X=None,
							   corner_labelsize=11,add_labeloffset=0,label_points=False):

	fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))

	tax1 = plot_labeled_ternary(scatter_comps, scatter_values,ax=ax1,label_points=label_points,
								corner_labelsize=corner_labelsize,add_labeloffset=add_labeloffset,vlim=vlim,cmap=cmap)

	tax2 = estimator_ternary_heatmap(hmap_scale,hmap_estimator,hmap_featurizer,hmap_feature_cols,scaler=hmap_scaler,use_X=hmap_use_X,
									 ax=ax2,labelsize=corner_labelsize,add_labeloffset=add_labeloffset,vlim=vlim,cmap=cmap)

	add_colorbar(fig,vlim=vlim,label='Max NH$_3$ Production Rate',cbrect=[0.9,0.2,0.02,0.7])

	return tax1,tax2	
	
def scatter_over_heatmap(scatter_comps, scatter_values,hmap_scale,hmap_estimator,vlim, ax=None, cmap=None,metric='median',
						hmap_featurizer=None,hmap_feature_cols=None,hmap_scaler=None,hmap_use_X=None,point_labeloffset=[0,0.02,0],
						corner_labelsize=11,add_labeloffset=0,marker='o',markersize=12,scatter_labels=None,scatter_labelsize=12):
	"""
		
	"""
	if ax is None:
		fig, ax = plt.subplots(figsize=(8,8))

	tax = estimator_ternary_heatmap(hmap_scale,hmap_estimator,hmap_featurizer,hmap_feature_cols,scaler=hmap_scaler,use_X=hmap_use_X,
									 ax=ax,labelsize=corner_labelsize,add_labeloffset=add_labeloffset,vlim=vlim,metric=metric,cmap=cmap)
								 
	if scatter_labels is None:
		# write blank labels
		scatter_labels = ['']*len(scatter_comps)
	
	points = [hmap_scale*get_coords_from_comp(c) for c in scatter_comps]
	
	for label, point, color_val in zip(scatter_labels,points,scatter_values):
		# must use tax.plot(); tax.scatter() does not work on top of heatmap
		if cmap is None:
			cmap = plt.get_cmap(plt.rcParams['image.cmap'])
		color = cmap((color_val-vlim[0])/(vlim[1]-vlim[0]))
		tax.plot([point],c=color,marker=marker,markersize=markersize,ls='',mec='white')
		tax.annotate('{}'.format(label),point+np.array(point_labeloffset)*hmap_scale,size=scatter_labelsize,ha='center',va='bottom',color='white')
		
	return tax
		
def draw_guidelines(tax,**line_kw):
	"""
	Add phase boundary lines and/or guidelines to ternary 
	
	Args:
		tax: ternary axes
		color: line color
		which: which lines to draw. Options: 'all','phase','guide'
		line_kw: kwargs for tax.line()
	"""
	
	ped_data = pd.read_csv('../data/BCA_PED_coords.csv',skipfooter=2,engine='python')
	for col in ['start','end']:
		ped_data[f'{col}_comp'] = ped_data[f'{col}'].map(lambda x: mg.Composition(BCA_formula_from_str(x)))
		ped_data[f'{col}_coords'] = ped_data[f'{col}_comp'].map(lambda x:get_coords_from_comp(x)*tax.get_scale())
	for i, row in ped_data[ped_data['draw_on_prodplot']==1].iterrows():
		tax.line(row['start_coords'],row['end_coords'],**line_kw)
	
	# ts = tax.get_scale()
		
	# #set up points
	# bca934 = np.array([0.19,0.25,0.56])*ts
	# b3a = np.array([0,.25,.75])*ts
	# c3a = np.array([.75,.25,0])*ts
	# ba = np.array([0,1/2,1/2])*ts
	# c = [ts,0,0]
	# b = [0,0,ts]
	
	
	# if which in ('all','guide'):
		# #guidelines
		# tax.line(b3a,c3a,ls=':',c=color,**line_kw)
		# tax.line(b,bca934,ls=':',c=color,**line_kw)
	# if which in ('all','phase'):
		# #phase boundaries
		# tax.line(c,ba,ls='--',c=color,**line_kw)
		# tax.line(ba,bca934,ls='--',c=color,**line_kw)
		# tax.line(bca934,c,ls='--',c=color,**line_kw)

