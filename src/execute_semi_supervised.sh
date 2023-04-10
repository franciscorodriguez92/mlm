#!/bin/bash
#chmod 777 ./execute_semi_supervised.sh
#nohup ./execute_semi_supervised.sh &> execute_semi_supervised.log &
source /data/frodriguez/venv_mlm/bin/activate
python -u predict_semi_supervised.py #&> experiment.log 2>&1