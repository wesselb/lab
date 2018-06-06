.PHONY: autodoc doc init test docopen clean testlocal

autodoc:
	rm -rf doc/source
	sphinx-apidoc -eMT -o doc/source/ lab
	rm doc/source/lab.rst
	pandoc --from=markdown --to=rst --output=doc/readme.rst README.md

doc:
	cd doc && make html

docopen:
	open doc/_build/html/index.html

init:
	pip install -r requirements.txt

test:
	python /usr/local/bin/nosetests tests --with-coverage --cover-html --cover-package=lab
	#
	# Run tests that cannot be run on CI.
	#
	python /usr/local/bin/nosetests tests.test:tf_bvn_cdf

clean:
	rm -rf .coverage cover
	rm -rf doc/_build doc/source doc/readme.rst
	find . | grep '\(\.DS_Store\|\.pyc\)$$' | xargs rm
