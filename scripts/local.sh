#!/bin/bash
if [ "$1" = -drop ]; then 
    echo Dropping database
    mysql -u userr2232 -h 127.0.0.1 < db/drop.sql
fi
if [ -f $(pwd)/.env ]; then
    echo Exporting env vars
    export $(cat .env | xargs)
fi
./credentials/cloud_sql_proxy -instances=deep-learning-308822:us-central1:hpo=tcp:3306 &
sleep 2
mysql -u userr2232 -h 127.0.0.1 < db/create.sql
python main.py root=$(pwd) action=run_study training.device=cpu hpo.ntrials=10