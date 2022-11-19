#!/usr/bin/env bash

inputFile="${1}"

# We want to edit the file if it hasn't already been patched
if ! grep -q "^currentDir=" ${inputFile}; then
    # Patch the prefix and installationdir. As far as I can tell, they're identical, so we set them identically here.
    sed -e 's|^prefix=.*|currentDir=$(realpath $(dirname \"$0\"))\nprefix=$(realpath ${currentDir}/..)|g' \
        -e 's|^installationdir=.*|installationdir=$(realpath ${currentDir}/..)|g'                         \
        $inputFile > $inputFile.out
    # Due to differences between sed on macOS and linux, we explicitly use a temp file
    mv $inputFile.out $inputFile
    # Moving removes the executable bit, so add it back
    chmod +x $inputFile
    echo "Successfully patched the install paths in fastjet-config"
fi
