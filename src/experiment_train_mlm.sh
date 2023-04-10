#!/bin/bash
#chmod 777 ./experiments.sh
#nohup ./experiments.sh &> experiment.log &
source /data/frodriguez/venv_mlm/bin/activate
python mlm_train.py
