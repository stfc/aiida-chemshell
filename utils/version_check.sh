#!/bin/bash 

GIT_TAG=$1
echo "Git tag: $GIT_TAG" 
APP_VERSION=$(python -c "
from aiida_chemshell import __version__
print(__version__) 
")
echo "Package version: $APP_VERSION" 

if [ "${GIT_TAG:1}" != "$APP_VERSION" ]; then 
  echo "ERROR: git tag does not match package version string" 
  exit 1 
fi 
