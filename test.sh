# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Runs CI tests on a local machine.
set -xeuo pipefail

# Install deps in a virtual env.
rm -rf _testing
rm -rf .pytype
mkdir -p _testing
readonly VENV_DIR="$(mktemp -d -p `pwd`/_testing rlax-env.XXXXXXXX)"
# in the unlikely case in which there was something in that directory
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
python3 -m pip install --upgrade pip uv
python3 -m uv pip install --upgrade pip setuptools wheel
python3 -m uv pip install --upgrade flake8 pytest-xdist pylint pylint-exit
python3 -m uv pip install --editable ".[test]"

# Install the requested JAX version
if [ "$JAX_VERSION" = "" ]; then
  : # use version installed in requirements above
elif [ "$JAX_VERSION" = "newest" ]; then
  pip install -U jax jaxlib
elif [ "$JAX_VERSION" = "nightly" ]; then
  pip install -U --pre jax jaxlib --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/
else
  pip install "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
fi

# Ensure rlax was not installed by one of the dependencies above,
# since if it is, the tests below will be run against that version instead of
# the branch build.
python3 -m uv pip uninstall rlax

# Lint with flake8.
python3 -m flake8 `find rlax -name '*.py' | xargs` --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.
PYLINT_ARGS="-efail -wfail -cfail -rfail"
# Download Google OSS config.
wget -nd -v -t 3 -O .pylintrc https://google.github.io/styleguide/pylintrc
# Enforce two space indent style.
sed -i "s/indent-string.*/indent-string='  '/" .pylintrc
# Append specific config lines.
echo "disable=unnecessary-lambda-assignment,no-value-for-parameter,use-dict-literal" >> .pylintrc
# Lint modules and tests separately.
pylint --rcfile=.pylintrc `find rlax -name '*.py' | grep -v 'test.py' | xargs` -d E1102|| pylint-exit $PYLINT_ARGS $?
# Disable `protected-access` warnings for tests.
pylint --rcfile=.pylintrc `find rlax -name '*_test.py' | xargs` -d W0212,E1130,E1102,E1120 || pylint-exit $PYLINT_ARGS $?
# Cleanup.
rm .pylintrc

# Build the package.
python3 -m uv pip install build
python3 -m build
python3 -m pip wheel --no-deps dist/rlax-*.tar.gz
python3 -m pip install rlax-*.whl

# Check types with pytype.
# Note: pytype does not support 3.11 as of 25.06.23
# See https://github.com/google/pytype/issues/1308
if [ `python -c 'import sys; print(sys.version_info.minor)'` -lt 11 ];
then
  python3 -m pip install pytype
  python3 -m pytype "rlax" -j auto --keep-going --disable import-error
fi;

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
cd _testing

# Main tests.
python3 -m pytest --numprocesses auto --pyargs rlax -k "not pop_art_test"

# Isolate tests that use `chex.set_n_cpu_device()`.
python3 -m pytest --numprocesses auto --pyargs rlax -k "pop_art_test"
cd ..

# Build Sphinx docs.
python3 -m uv pip install --editable ".[docs]"
cd docs && make html
cd ..

# cleanup
rm -rf _testing

set +u
deactivate
echo "All tests passed. Congrats!"
