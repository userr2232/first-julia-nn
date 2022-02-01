#!/bin/bash
if [ "$1" = -drop ]; then 
    echo Dropping database
    mysql < db/drop.sql
fi
mysql < db/create.sql
python main.py root=$(pwd) action=run_study training.device=cpu hpo.ntrials=1 hpo.max_nlayers=5 hpo.max_nunits=20 training.epochs=10