DOC_BUILD_DIR=build/sphinx
DOC_SOURCE_DIR=docs
SOURCE_DIR=libspn

# Make sure the targets are never "up to date"
.PHONY: build

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

build:
	@python3 setup.py build

clean:
	@python3 setup.py clean

install:
	@pip3 install --user .

# Run build after install since dev-install doesn't do it
# Run it after install, so that deps are installed
dev-install:
	@pip3 install --user -e .
	$(MAKE) build
