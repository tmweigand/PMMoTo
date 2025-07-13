#!/bin/bash
set -e

# Usage: ./release.sh <version>
# Example: ./release.sh 0.1.1

if [ -z "$1" ]; then
  echo "Error: Version number required."
  echo "Usage: $0 <version>"
  exit 1
fi

VERSION=$1

echo "Cleaning previous build artifacts..."
rm -rf build dist *.egg-info

echo "Updating version to $VERSION in pyproject.toml..."
sed -i.bak -E "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
rm pyproject.toml.bak

echo "Building distributions..."
python -m build

echo "Uploading to PyPI..."
python -m twine upload dist/*

echo "Done! Version $VERSION released."

