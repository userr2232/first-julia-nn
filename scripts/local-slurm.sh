#!/bin/bash
if [ "$1" = -drop ]; then 
    echo Dropping database
    mysql -u userr2232 < db/drop.sql
fi
mysql -u userr2232 < db/create.sql
python main.py root=$(pwd) action=run_study training.device=cpu hpo.ntrials=1 hpo.max_nlayers=5 hpo.max_nunits=20 training.epochs=10
