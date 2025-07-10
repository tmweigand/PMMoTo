#!/bin/bash
# Generate API documentation and build HTML docs for PMMoTo

set -euo pipefail

DOCS_DIR="docs"
API_DOCS_DIR="${DOCS_DIR}/api"
SRC_DIR="src/pmmoto"

echo "Cleaning previous API docs in ${API_DOCS_DIR}..."
rm -fv "${API_DOCS_DIR}"/pmmoto.*.rst "${API_DOCS_DIR}"/modules.rst || true

echo "Generating new API documentation with sphinx-apidoc into ${API_DOCS_DIR}..."
sphinx-apidoc -o "${API_DOCS_DIR}" "${SRC_DIR}" --force --module-first

echo "Building HTML documentation..."
make -C "${DOCS_DIR}" clean html

echo "Documentation build completed successfully!"
