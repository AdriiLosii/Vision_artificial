#!/bin/bash

wget https://www.filepicker.io/api/file/VQKdmzKqSLWruVhx7Xdd -O INRIAPerson.zip
unzip -q INRIAPerson.zip -d data/images
rm -f INRIAPerson.zip
