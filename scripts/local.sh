#!/bin/bash
if [[ -f $(pwd)/.env ]]; then
    echo Exporting env vars
    export $(cat .env | xargs)
fi
./credentials/cloud_sql_proxy -instances=deep-learning-308822:us-central1:hpo=tcp:3306 &
sleep 30
if [[ "$1" = -drop ]]; then 
    echo Dropping database
    mysql -u userr2232 -h 127.0.0.1 < db/drop.sql
fi
mysql -u userr2232 -h 127.0.0.1 < db/create.sql
sleep 30
for i in {1..4}; do
    python main.py root=$(pwd) action=run_study training.device=cpu hpo.ntrials=300 &
    if [[ $i = 1 ]]; then
        sleep 10
    fi
done
wait

