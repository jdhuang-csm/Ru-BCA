# Tools for model testing and evaluation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

def multi_min(arrays):
	"""
	Return the minimum scalar value of multiple arrays
	
	Args:
		arrays: list of numpy arrays
	"""
	mins = []
	for arr in arrays:
		mins.append(np.min(arr))
	return min(mins)

def multi_max(arrays):
	"""
	Return the maximum scalar value of multiple arrays
	
	Args:
		arrays: list of numpy arrays
	"""
	maxs = []
	for arr in arrays:
		maxs.append(np.max(arr))
	return max(maxs)

class repeating_KFold():
	"""
	KFold splitter that performs multiple independent splits of the dataset. For use with sklearn and mlxtend functions/classes that take a splitter object
	Intended for use with shuffle=True to reduce bias for one particular train-test split
	
	Args:
		repeat: int, number of times to repeat
		n_splits: number of splits
		shuffle: if True, shuffle dataset before splitting
		random_state: specify a random state for shuffle
	"""
	def __init__(self,repeat,n_splits,shuffle=True,random_state=None):
		self.repeat = repeat
		self.n_splits = n_splits
		self.shuffle = shuffle
		self.random_state = random_state
		self.kf = KFold(n_splits,shuffle)
		# set seeds for consistency if random state specified
		if self.random_state is not None:
			r = np.random.RandomState(self.random_state)
			self.seeds = r.choice(np.arange(0,repeat*10,1),self.repeat,replace=False)
		else:
			self.seeds = [None]*self.repeat
		
	def split(self,X,y=None,groups=None):
		for n,seed in zip(range(self.repeat),self.seeds):
			self.kf.random_state = seed
			for train,test in self.kf.split(X,y,groups):
				yield train,test
				
	def get_n_splits(self,X=None,y=None,groups=None):
		return self.n_splits*self.repeat
	
def KFold_cv(estimator,X,y,sample_weight=None,n_splits=5,pipeline_learner_step='auto',random_state=None):
	"""
	Perform k-fold cross-validation
	
	Args:
		estimator: sklearn estimator instance
		X: data matrix (nxm)
		y: response (n-vector)
		sample_weight: weights for fitting data. If None, defaults to equal weights
		n_splits: number of folds. Default 5
		pipeline_learner_step: if estimator is a Pipeline instance, index of the learner step
		random_state: random state for KFold shuffle
	Returns:
		 actual: acutal y values for test folds
		 pred: predicted y values for test folds
		 train_scores: list of training r2 scores
		 test_scores: list of test r2 scores
	"""
	if random_state is not None:
		kf = KFold(n_splits,shuffle=True,random_state=random_state)
	else:
		kf = KFold(n_splits,shuffle=True)
	if len(X)!=len(y):
		raise ValueError('X and y must have same first dimension')
	
	# if y is pandas series, convert to array. No info required from Series object
	if type(y)==pd.core.series.Series:
		y = y.values
	
	train_scores = np.empty(n_splits)
	test_scores = np.empty(n_splits)
	actual = np.zeros_like(y)
	pred = np.zeros_like(y)
	for i, (train_index,test_index) in enumerate(kf.split(X)):
		if type(X)==pd.core.frame.DataFrame:
			X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
		else:
			X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		if sample_weight is not None:
			w_train = sample_weight[train_index]
			w_test = sample_weight[test_index]
		
		if sample_weight is not None:
			if type(estimator)==Pipeline:
				#if estimator is a Pipeline, need to specify name of learning step in fit_params for sample_weight
				if pipeline_learner_step=='auto':
					# determine which step is the learner based on existence of _estimator_type attribute
					step_objects  = [step[1] for step in estimator.steps]
					objdirs = [dir(obj) for obj in step_objects]
					learner_idx = np.where(['_estimator_type' in d for d in objdirs])[0]
					if len(learner_idx)==1:
						pipeline_learner_step = learner_idx[0]
					else:
						raise Exception("Can''t determine pipeline_learner_step. Must specify manually")
				est_name = estimator.steps[pipeline_learner_step][0]
				estimator.fit(X_train,y_train,**{f'{est_name}__sample_weight':w_train})
			else:
				estimator.fit(X_train,y_train,sample_weight=w_train)
			train_scores[i] = estimator.score(X_train,y_train,sample_weight=w_train)
			test_scores[i] = estimator.score(X_test,y_test,sample_weight=w_test)
		else:
			# not all estimators' fit() methods accept sample_weight arg - can't just pass None
			estimator.fit(X_train,y_train)
			train_scores[i] = estimator.score(X_train,y_train)
			test_scores[i] = estimator.score(X_test,y_test)
		actual[test_index] = y_test
		pred[test_index] = estimator.predict(X_test)
	
	return actual, pred, train_scores, test_scores

def repeated_KFold_cv(estimator,X,y,repeat,sample_weight=None,n_splits=5,pipeline_learner_step='auto',random_state=None):
	"""
	Perform k-fold cross-validation with multiple random splits
	
	Args:
		estimator: sklearn estimator instance
		X: data matrix (nxm)
		y: response (n-vector)
		repeat: number of times to repeat KFold CV
		sample_weight: weights for fitting data. If None, defaults to equal weights
		n_splits: number of folds. Default 5
		pipeline_learner_step: if estimator is a Pipeline instance, index of the learner step
		random_state: random state for KFold shuffle
	Returns:
		 actuals: list of actual y vectors for all CV repetitions
		 preds: list of predicted y vectors for all CV repetitions
		 agg_test_scores: list of r2 scores for all CV repetitions
		 agg_test_maes: list of MAEs for all CV repetitions
	"""
	actuals = np.empty((repeat,len(y)))
	preds = np.empty_like(actuals)
	agg_test_scores = np.empty(repeat)
	agg_test_maes = np.empty(repeat)
	
	# set seeds for consistency if specified
	if random_state is not None:
		r = np.random.RandomState(random_state)
		seeds = r.choice(np.arange(0,repeat*10,1),repeat,replace=False)
	else:
		seeds = [None]*repeat
		
	for n in range(repeat):
		act,pred,train,test = KFold_cv(estimator,X,y,sample_weight,n_splits,pipeline_learner_step,random_state=seeds[n])
		agg_test_score = r2_score(act,pred,sample_weight=sample_weight)
		agg_mae = mean_absolute_error(act,pred,sample_weight=sample_weight)
		actuals[n] = act
		preds[n] = pred
		agg_test_scores[n] = agg_test_score
		agg_test_maes[n] = agg_mae

	return actuals, preds, agg_test_scores, agg_test_maes
	
def KFold_pva(estimator,X,y,sample_weight=None,n_splits=5,random_state=None,ax=None,show_metrics=['r2','mae'],text_kw={},logscale=False,s=10,line_kw={'c':'g'},**scatter_kw):
	"""
	Perform k-fold cross-validation and plot predicted vs. actual for test set
	
	Args:
		estimator: sklearn estimator instance
		X: data matrix (nxm)
		y: response (n-vector)
		sample_weight: vector of sample weights. If None, equal weights assigned
		n_splits: number of folds. Default 5
		random_state: random state for KFold shuffle
		ax: axis on which to plot
		show_metrics: list of metrics to calculate and annotate on plot. Options: 'r2', 'mae'
		text_kw: kwargs for metric text; passed to plt.text()
		logscale: if True, plot as log-log 
		s: marker size
		line_kw: kwargs for ideal x=y line
		scatter_kw: kwargs to pass to plt.scatter()
		
	Returns: 
		train_scores: k-array of train scores
		test_scores: k-array of test scores
		agg_test_score: overall test score (r2) considering all test folds together 
	"""
	y, y_pred, train_scores, test_scores = KFold_cv(estimator,X,y,n_splits=n_splits,random_state=random_state)
	agg_test_score = r2_score(y,y_pred,sample_weight=sample_weight)
	
	ax = pred_v_act_plot(y,y_pred,sample_weight,ax,show_metrics,text_kw,logscale,s,line_kw,**scatter_kw)
	
	return train_scores, test_scores, agg_test_score
	
def repeated_KFold_pva(estimator,X,y,repeat,plot_type='series',sample_weight=None,n_splits=5,pipeline_learner_step=1,random_state=None,
					ax=None,show_metrics=['r2','mae'],text_kw={},logscale=False,s=10,line_kw={'c':'g'},**scatter_kw):
	"""
	Perform k-fold cross-validation and plot predicted vs. actual for test set
	
	Args:
		estimator: sklearn estimator instance
		X: data matrix (nxm)
		y: response (n-vector)
		repeat: number of times to repeat KFold CV
		sample_weight: weights for fitting data. If None, defaults to equal weights
		n_splits: number of folds. Default 5
		pipeline_learner_step: if estimator is a Pipeline instance, index of the learner step
		random_state: random state to determine random seeds for KFold shuffles
		ax: axis on which to plot
		show_metrics: list of metrics to calculate and annotate on plot. Options: 'r2', 'mae'
		text_kw: kwargs for metric text; passed to plt.text()
		logscale: if True, plot as log-log 
		s: marker size
		scatter_kw: kwargs to pass to plt.scatter()
		
	Returns: 
		train_scores: k-array of train scores
		test_scores: k-array of test scores
		tot_test_score: overall test score (r2) considering all test folds together 
	"""
	actuals, preds, agg_test_scores, agg_test_maes = repeated_KFold_cv(estimator,X,y,repeat,sample_weight,n_splits,pipeline_learner_step,random_state)
	
	if plot_type=='series':
		# plot each repetition as a separate series
		for y, y_pred in zip(actuals, preds):
			ax = pred_v_act_plot(y,y_pred,sample_weight,ax,show_metrics=None,text_kw=text_kw,logscale=logscale,s=s,line_kw=line_kw,**scatter_kw)
	elif plot_type=='mean':
		# average predicted values for each point across repetitions
		y = np.mean(actuals,axis=0)
		y_pred = np.mean(preds,axis=0)
		ax = pred_v_act_plot(y,y_pred,sample_weight,ax,show_metrics=None,text_kw=text_kw,logscale=logscale,s=s,line_kw=line_kw,**scatter_kw)

	# metrics need to be aggregated across repetitions
	metric_txt = ''
	for metric in show_metrics:
		if metric=='r2':
			metric_txt += '$r^2: \ {}$\n'.format(round(np.mean(agg_test_scores),3))
		elif metric=='mae':
			mae_scale = int(np.ceil(np.log10(np.mean(agg_test_maes))))
			if mae_scale < 3:
				mae_round = 3 - mae_scale
			else:
				mae_round = 0
			metric_txt += 'MAE: {}\n'.format(round(np.mean(agg_test_maes),mae_round))
			
	if len(metric_txt) > 0:
		x = text_kw.pop('x',0.05)
		y = text_kw.pop('y',0.95)
		ax.text(x,y,metric_txt,transform=ax.transAxes,va='top',**text_kw)

def plot_pva(estimator,X,y,sample_weight=None,ax=None,show_metrics=['r2','mae'],text_kw={},logscale=False,s=10,line_kw={'c':'g'},**scatter_kw):
	"""
	Plot predicted vs. actual for fitted estimator

	Args:
		estimator: fitted sklearn estimator instance
		X: data matrix (nxm)
		y: response (n-vector)
		sample_weight: sample weights. Only used to calculate metrics (r2, mae)
		ax: axis on which to plot
		show_metrics: list of metrics to calculate and annotate on plot. Options: 'r2', 'mae'
		text_kw: kwargs for metric text; passed to plt.text()
		logscale: if True, plot as log-log 
		s: marker size
	"""
	y_pred = estimator.predict(X)
	ax = pred_v_act_plot(y,y_pred,sample_weight,ax,show_metrics,text_kw,logscale,s,line_kw,**scatter_kw)
	
def pred_v_act_plot(y,y_pred,sample_weight=None,ax=None,show_metrics=['r2','mae'],text_kw={},logscale=False,s=10,line_kw={'c':'g'},legend=True,**scatter_kw):
	"""
	Plot predicted vs. actual

	Args:
		y: actual values
		y_pred: predictions
		sample_weight: sample weights. Only used to calculate metrics (r2, mae)
		ax: axis on which to plot
		show_metrics: list of metrics to calculate and annotate on plot. Options: 'r2', 'mae'
		text_kw: kwargs for metric text; passed to plt.text()
		logscale: if True, plot as log-log 
		s: marker size
	"""
	if ax is None:
		fig, ax = plt.subplots()
	
	axmin = multi_min([y,y_pred])
	axmax = multi_max([y,y_pred])
	if logscale==False:
		ax.scatter(y,y_pred,s=s,**scatter_kw)
		ax.plot([axmin,axmax],[axmin,axmax],**line_kw,label='Ideal')
	elif logscale==True:
		ax.loglog(y,y_pred,'o',markersize=s,**scatter_kw)
		ax.loglog([axmin,axmax],[axmin,axmax],**line_kw,label='Ideal')
		
	metric_txt = ''
	if show_metrics is not None:
		for metric in show_metrics:
			if metric=='r2':
				r2 = r2_score(y,y_pred,sample_weight=sample_weight)
				metric_txt += '$r^2: \ {}$\n'.format(round(r2,3))
			elif metric=='mae':
				test_mae = mean_absolute_error(y,y_pred,sample_weight=sample_weight)
				mae_scale = int(np.ceil(np.log10(test_mae)))
				if mae_scale < 3:
					mae_round = 3 - mae_scale
				else:
					mae_round = 0
				metric_txt += 'MAE: {}\n'.format(round(test_mae,mae_round))
			
	if len(metric_txt) > 0:
		x = text_kw.pop('x',0.05)
		y = text_kw.pop('y',0.95)
		ax.text(x,y,metric_txt,transform=ax.transAxes,va='top',**text_kw)
	
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
	if legend:
		ax.legend(loc='lower right')
	
	return ax
	
	
class GridSearchRepeatedCV():
	def __init__(self,estimator,param_grid):
		self.estimator = deepcopy(estimator)
		self.param_grid = param_grid
		
	def fit(self,X,y,repeat,sample_weight=None,n_splits=5,pipeline_learner_step='auto',random_state=None):
		meshgrid = np.meshgrid(*self.param_grid.values())
		self.param_meshgrid_ = dict(zip(self.param_grid.keys(),meshgrid))
		self.grid_scores_ = np.zeros_like(meshgrid[0],dtype='float')
		self.grid_params_ = np.empty_like(meshgrid[0],dtype='object')
		# iterate over parameter combinations
		for idx, tmpvalue in np.ndenumerate(meshgrid[0]):
			# get parameter values and set
			params = {}
			for param_name, param_array in zip(self.param_grid.keys(),meshgrid):
				params[param_name] = param_array[idx]
			self.estimator.set_params(**params)
			# perform cv
			y_act, y_pred, agg_scores, agg_maes = repeated_KFold_cv(self.estimator,X,y,repeat,sample_weight,n_splits,pipeline_learner_step,random_state)
			self.grid_scores_[idx] = np.mean(agg_scores)
			self.grid_params_[idx] = params
			
		# get best index
		self.best_index_ = np.argmax(self.grid_scores_)
		self.best_params_ = self.grid_params_.ravel()[self.best_index_]
		self.best_score_ = np.max(self.grid_scores_)
		
	@property
	def grid_results_(self):
		scores = self.grid_scores_.ravel()
		params = self.grid_params_.ravel()
		results = [(p,s) for p,s in zip(params,scores)]
		return results
		
	@property
	def ranked_results(self):
		results = self.grid_results_
		ranked_results = sorted(results,key=lambda x: x[1],reverse=True)
		ranked_results = [(i+1,p,s) for i,(p,s) in enumerate(ranked_results)]
		return ranked_results
		
	@property
	def result_df(self):
		scores = self.grid_scores_.ravel()
		params = self.grid_params_.ravel()
		df = pd.DataFrame(list(params))
		df['score'] = scores
		return df
		
	def plot_grid_results(self,ax=None,fixed_params={},mark_best=True,colorbar=True,**scatter_kw):
		filter_meshgrid = self.param_meshgrid_.copy()
		filter_param = self.grid_params_.copy()
		filter_score = self.grid_scores_.copy()
		for param, value in fixed_params.items():
			idx = np.where(filter_meshgrid[param]==value)
			for p in filter_meshgrid.keys():
				filter_meshgrid[p] = filter_meshgrid[p][idx]
			filter_param = filter_param[idx]
			filter_score = filter_score[idx]
		
		plot_params = [p for p in self.param_grid.keys() if p not in fixed_params.keys()]
		if len(plot_params) > 3:
			raise Exception('Too many free parameters to plot')
		param_arrays = []
		for param in plot_params:
			param_arrays.append([gp[param] for gp in filter_param.ravel()])
		scores = filter_score.ravel()
		
		if ax is None and len(plot_params) < 3:
			fig, ax = plt.subplots()
		elif ax is None and len(plot_params)==3:
			fig = plt.figure()
			ax = fig.add_subplot(111,projection='3d')
		else:
			fig = ax.get_figure()
			
		if len(plot_params)==1:
			ax.scatter(param_arrays[0],scores,**scatter_kw)
			ax.set_xlabel(plot_params[0])
			ax.set_ylabel('CV Score')
			if mark_best:
				# mark the best parameter value with a vertical line
				best_param = param_arrays[0][np.argmax(scores)]
				ax.axvline(best_param,c='k',lw=1)
		elif len(plot_params)==2:
			p = ax.scatter(param_arrays[0],param_arrays[1],c=scores,**scatter_kw)
			ax.set_xlabel(plot_params[0])
			ax.set_ylabel(plot_params[1])
			if colorbar:
				fig.colorbar(p,ax=ax,label='CV Score')
			if mark_best:
				# outline the best point in red
				best_idx = np.argmax(scores)
				ax.scatter(param_arrays[0][best_idx],param_arrays[1][best_idx],facecolors='none',edgecolors='r',**scatter_kw)
		elif len(plot_params)==3:
			p = ax.scatter(param_arrays[0],param_arrays[1],param_arrays[2],c=scores,**scatter_kw)
			ax.set_xlabel(plot_params[0])
			ax.set_ylabel(plot_params[1])
			ax.set_zlabel(plot_params[2])
			if colorbar:
				fig.colorbar(p,ax=ax,label='CV Score')
			if mark_best:
				best_idx = np.argmax(scores)
				ax.scatter(param_arrays[0][best_idx],param_arrays[1][best_idx],param_arrays[2][best_idx],facecolors='none',edgecolors='r',**scatter_kw)
			
		return ax