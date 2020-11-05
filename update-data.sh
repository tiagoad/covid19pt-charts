#!/bin/bash

mkdir -p vendor
cd vendor

new_data=0

function download() {
	curl -s "$1" > "$2.new"

	if ! diff "$2" "$2.new" 2>/dev/null >/dev/null
	then
		new_data=1
	fi

	mv "$2.new" "$2"
}

# DSSG
download https://raw.githubusercontent.com/dssg-pt/covid19pt-data/master/data.csv dssg_data.csv
download https://raw.githubusercontent.com/dssg-pt/covid19pt-data/master/amostras.csv dssg_samples.csv

if [[ $new_data == 1 ]]; then
	echo "Data changed"
	exit 0
else
	echo "No updates to data"
	exit 1
fi