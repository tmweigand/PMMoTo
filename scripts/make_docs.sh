#!/bin/bash
# Generate API documentation and build HTML docs for PMMoTo

set -e

# Clean previous API docs (optional, keeps docs clean)
rm -rf docs/pmmoto.*.rst docs/modules.rst

# Generate API documentation
sphinx-apidoc -o docs/ src/pmmoto --force --module-first

# Build HTML documentation
make -C docs clean html