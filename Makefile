DOC_BUILD_DIR=build/sphinx
DOC_SOURCE_DIR=docs
SOURCE_DIR=libspn

flake8:
	@python3 setup.py flake8

doc:
	@python3 setup.py build_sphinx

doc-auto:
	@python3 -c "import sphinx_autobuild; sphinx_autobuild.main()" -z $(SOURCE_DIR) -d $(DOC_BUILD_DIR)/doctrees $(DOC_SOURCE_DIR) $(DOC_BUILD_DIR)/html

test:
	@python3 setup.py test

wheel:
	@python3 setup.py bdist_wheel

sdist:
	@python3 setup.py sdist

install:
	@pip3 install --user .

dev-install:
	@pip3 install --user -e .

build:
	@python3 setup.py build
