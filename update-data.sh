#!/bin/bash

if cd vendor/covid19pt-data/ && git status; then
  git pull
else
  git clone https://github.com/dssg-pt/covid19pt-data.git vendor/covid19pt-data
fi
