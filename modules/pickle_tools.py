import pickle

def save_pickle(obj, file):
	with open(file,'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
	print('Dumped pickle to {}'.format(file))
	
def load_pickle(file):
	with open(file,'rb') as f:
		return pickle.load(f)
	