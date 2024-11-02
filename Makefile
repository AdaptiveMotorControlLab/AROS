AROS_VERSION := 0.0.1rc1

dist:
	python3 -m pip install virtualenv
	python3 -m pip install --upgrade build twine
	python3 -m build --wheel --sdist

build: dist

archlinux:
	mkdir -p dist/arch
	cp PKGBUILD dist/arch
	cp dist/aros_node-${AROS_VERSION}.tar.gz dist/arch
	(cd dist/arch; makepkg --skipchecksums -f)
