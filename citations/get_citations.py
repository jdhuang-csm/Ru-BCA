# slight modification of makecite.cmdline code to avoid some errors and make the citations even though some packages can't be found
import glob
import os
import sys
import makecite
from makecite.discover import find_all_files
from parsers import parser_map
from makecite.cmdline import get_bibtex_from_package, get_bibtex_from_citation_file, get_bibtex
from argparse import ArgumentParser, RawTextHelpFormatter
import re

_bib_path = os.path.join(makecite.__path__[0],
						 'bibfiles')
						 
cite_tag_pattr = re.compile('@[a-zA-Z]+\{(.*),')

module_path = 'C:\\Users\\jdhuang\\OneDrive - Colorado School of Mines\\Research\\Manuscripts\\Cadigan_B2CA_2019\\for_publication\\modules'
	
def get_all_packages(paths, extensions=['.py', '.ipynb'],
					 include_imported_dependencies=False):
	"""Get a unique list (set) of all package names imported by all files of
	the requested extensions
	Parameters
	----------
	paths : list, str
	extensions : list, iterable
	include_imported_dependencies : bool, optional
	Returns
	-------
	packages : set
	"""
	if isinstance(paths, str):
		paths = [paths]

	all_packages = set()
	for path in paths:
		if os.path.isfile(path):
			basename, ext = os.path.splitext(path)
			file_dict = {ext: [path]}

		else:
			file_dict = find_all_files(path, extensions=extensions)

		for ext, files in file_dict.items():
			if ext not in parser_map:
				raise ValueError('File extension "{0}" is not supported.'
								 .format(ext))

			for file in files:
				_packages = parser_map[ext](file)
				all_packages = all_packages.union(_packages)

	if include_imported_dependencies:
		init_modules = sys.modules.copy()

		# Now we have a list of package names, so we can import them and track
		# what other packages are imported as dependencies. If requested, we add
		# those to the package list as well
		for package_name in all_packages:
			try:
				importlib.import_module(package_name)
			except ImportError:
				# here, just skip if we can't import: a warning is issued later
				pass

		loaded_modules = sys.modules.copy()
		diff_modules = set(loaded_modules.keys()) - set(init_modules.keys())

		additional_modules = set()
		for module in diff_modules:
			top_level = module.split('.')[0]

			if top_level.startswith('_'):
				continue

			additional_modules.add(top_level)

		all_packages = all_packages.union(additional_modules)

	return all_packages

def get_installed_packages(paths,module_paths=[],extensions=['.py','.ipynb'],include_imported_dependencies=False):

	module_files = []
	for mpath in module_paths:
		module_files += [os.path.basename(mfile) for mfile in glob.glob(os.path.join(mpath,'*.py'))]
	
	packages = get_all_packages(paths=paths,
								extensions=extensions,
								include_imported_dependencies=include_imported_dependencies)
								
	# remove local module files
	packages -= set([mfile.replace('.py','') for mfile in module_files])

	return packages

parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
	
parser.add_argument('-e', '--ext', action='append', dest='extensions',
					default=None,
					help='Specify the file extensions to look for and '
						 'parse. Currently, only .py and .ipynb are '
						 'supported.')

parser.add_argument('-o', '--output-file', dest='output_file',
					default=None,
					help='For example, "software-refs.bib". The file to '
						 'save the bibtex references to. If the file '
						 'exists, this will append the citations to the ' 'end of the existing file. Otherwise, the file '
						 'is created.')

parser.add_argument('-r', '--recursive', dest='recursive', default=False,
					action='store_true',
					help='Discover what packages are imported by imports '
						  'in the script or modules being parsed. For '
						  'example, if the script you run makecite on '
						  'imports another package, and you want to cite '
						  'all packages used, use this flag.')

parser.add_argument('--aas', action='store_true', dest='aas_tag',
					default=False,
					help='Also generate a AAS Latex \software{} tag with '
						 'all packages used.')

parser.add_argument('paths', type=str, nargs='+',
					help='A path, filename, or list of paths to search '
						 'for imported packages.')

# parser.add_argument('--version', action='version',
					# version=__version__)


args = parser.parse_args()

if not args.extensions:
	args.extensions = ['.py', '.ipynb']

packages = get_installed_packages(paths=args.paths,
							module_paths=[module_path],
							extensions=args.extensions,
							include_imported_dependencies=args.recursive)

all_bibtex = ""
y_citation = []
n_citation = []
name_to_tags = dict()
for package in sorted(list(packages)):
	try:
		bibtex = get_bibtex_from_package(package)
		if bibtex is None:
			try:
				bibtex = get_bibtex_from_citation_file(package)
			except Exception:
				# AttributeError: if module doesn't have __path__ attribute (core packages, e.g. copy)
				# ModuleNotFoundError: if module not installed (imported from other path)
				pass
		if bibtex is None:
			bibtex = get_bibtex(package)
		y_citation.append(package)
		name_to_tags[package] = cite_tag_pattr.findall(bibtex)
	except ValueError:
		# Package doesn't have a .bib file in this repo. For now, just alert
		# the user, but we might want to try a web query or something?
		n_citation.append(package)
		continue

	all_bibtex = "{0}\n{1}".format(all_bibtex, bibtex)

# print out some information about the packages identified, and ones
# that don't have citation information
print("Packages detected with citation information:")
print("\t{0}".format(", ".join(y_citation)))

print("\nPackages with no citation information:")
print("\t{0}".format(", ".join(n_citation)))

if args.output_file:
	# save .bib output file
	print("\nBibtex file generated: {0}".format(args.output_file))

	with open(args.output_file, 'a') as f:
		f.write(all_bibtex)

else:
	print("\nBibtex:")
	print(all_bibtex)

if args.aas_tag:
	cites = []
	for name, tags in name_to_tags.items():
		cites.append('{0} \\citep{{{1}}}'.format(name, ', '.join(tags)))

	software = r'\software{{{0}}}'.format(', '.join(cites))

	print("\nSoftware tag for AAS journals:")
	print(software)
		