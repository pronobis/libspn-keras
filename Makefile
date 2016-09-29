DOC_BUILD_DIR=docs/build
DOC_SOURCE_DIR=docs/source
DOC_BUILD_DEV_DIR=docs/build-dev
DOC_SOURCE_DEV_DIR=docs/source-dev
SOURCE_DIR=libspn

define check-python=
@if [ `python -c 'import sys; print(sys.version_info[0])'` = "3" ];\
then\
echo "Using Python 3";\
else\
echo "You are using Python 2, that means trouble!";\
exit 1;\
fi
endef

kot:

flake8:
	$(check-python)
	@flake8 libspn

doc:
	$(check-python)
	sphinx-build -b html -d $(DOC_BUILD_DIR)/doctrees $(DOC_SOURCE_DIR) $(DOC_BUILD_DIR)/html
	@echo
	@echo "Documentation build finished! Generated documentation is in $(DOC_BUILD_DIR)/html."

doc-dev:
	$(check-python)
	sphinx-build -b html -d $(DOC_BUILD_DEV_DIR)/doctrees $(DOC_SOURCE_DEV_DIR) $(DOC_BUILD_DEV_DIR)/html
	@echo
	@echo "Documentation build finished! Generated documentation is in $(DOC_BUILD_DEV_DIR)/html."

doc-auto:
	$(check-python)
	sphinx-autobuild -z $(SOURCE_DIR) -d $(DOC_BUILD_DIR)/doctrees $(DOC_SOURCE_DIR) $(DOC_BUILD_DIR)/html

doc-dev-auto:
	$(check-python)
	sphinx-autobuild -z $(SOURCE_DIR) -d $(DOC_BUILD_DEV_DIR)/doctrees $(DOC_SOURCE_DEV_DIR) $(DOC_BUILD_DEV_DIR)/html

test:
	$(check-python)
	@nose2 -v
	@echo "Tests finished!"
