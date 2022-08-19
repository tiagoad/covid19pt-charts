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
download https://raw.githubusercontent.com/dssg-pt/covid19pt-data/master/amostras.csv dssg_amostras.csv
download https://raw.githubusercontent.com/dssg-pt/covid19pt-data/master/vacinas.csv dssg_vacinas.csv
download https://raw.githubusercontent.com/dssg-pt/covid19pt-data/master/dados_diarios.csv dssg_dados_diarios.csv
#download https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/Portugal.csv owid_vaccines.csv

if [[ $new_data == 1 ]]; then
	echo "Data changed"
	exit 0
else
	echo "No updates to data"
	exit 1
fi
