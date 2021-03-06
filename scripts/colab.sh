#!/bin/bash
if [ "$1" = -drop ]; then 
    echo Dropping database
    mysql -u userr2232 -h 127.0.0.1 < $(pwd)/first-julia-nn/db/drop.sql
fi
mysql -u userr2232 -h 127.0.0.1 < $(pwd)/first-julia-nn/db/create.sql
python $(pwd)/first-julia-nn/main.py root=$(pwd)/first-julia-nn action=create_partitions
python $(pwd)/first-julia-nn/main.py root=$(pwd)/first-julia-nn action=run_study hpo.ntrials=50