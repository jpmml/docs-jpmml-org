from setuptools import find_packages, setup

setup (
	name = "docs_jpmml_org",
	version = "0.1.0",
	description = "Python library for https://docs.jpmml.org notebooks",
	author = "Villu Ruusmann",
	author_email = "villu.ruusmann@gmail.com",
	license = "GNU Affero General Public License (AGPL) version 3.0",

	packages = find_packages(),
	python_requires = ">=3.8",
	install_requires = [
		"ucimlrepo==0.0.7",
	]
)