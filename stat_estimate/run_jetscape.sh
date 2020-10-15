#!/usr/bin/env bash

userConfig="$1"
outputDir=$(dirname "$1")

echo "User config: ${userConfig}"

cd ${outputDir}

# We need the LBT-tables symlink so it will actually run properly...
if [ ! -L "${outputDir}/LBT-tables" ]; then
    # This is super roundabout, but it works, so whatever...
    ln -s /home/jetscape-user/LBT-tables .
fi

runJetscape ${userConfig} /home/jetscape-user/JETSCAPE/config/jetscape_master.xml
