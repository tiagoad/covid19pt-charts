This code generates a few charts out of the dggs-pt's covid19pt-data repository, which contains all the data available in the official reports, in a friendly CSV format.

The charts are automatically deployed to https://covid19.tdias.pt/ when the DSSG repository is updated.

To clone/update the data, run `./update-data.sh`.

Install poetry globally (`pip install poetry`), then run `poetry install` in the project directory to install all the python dependencies in a virtualenv.

Run `poetry shell` to enter a virtualenv shell, and run `python chart.py` to generate the charts in your project directory as PNG pictures. Alternatively, run `poetry run python ./chart.py`)

This script is really not well written nor production code. I did this on a few weekends seeking just some nice charts. Hope you like them.