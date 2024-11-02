#!/bin/bash

# Re-install the package. By running './reinstall.sh'
#
# Note that AROS uses the build
# system specified in
# PEP517 https://peps.python.org/pep-0517/ and
# PEP518 https://peps.python.org/pep-0518/
# and hence there is no setup.py file.

set -e # abort on error

pip uninstall -y aros-node

# Get version
VERSION=0.0.1rc1
echo "Upgrading to AROS v${VERSION}"

# Upgrade the build system (PEP517/518 compatible)
python3 -m pip install virtualenv
python3 -m pip install --upgrade build
python3 -m build --sdist --wheel .

# Reinstall the package with most recent version
pip install --upgrade --no-cache-dir "dist/aros_node-${VERSION}-py3-none-any.whl"
