Prepare preprocessing instance:
mkdir domain_adaption
python3 -m venv env_data_processing
source env_data_processing/bin/activate
#pip3 install -r requirements.txt
python3 -m pip install numpy pandas torch transformers datasets unidecode sklearn
sudo mkdir data
cd data
aws s3 cp s3://metwo-unlabeled-data/corpus/ . --recursive
mkdir /src
cd src
python3 mlm_data_preprocessing.py
sudo shutdown -h now

#####################################################
#####################################################
# EXPERIMENTS PREPROCESSING:::
#!/bin/bash
#chmod 777 ./experiments.sh
#nohup ./experiments.sh &> experiment.log &
source ../env_data_processing/bin/activate
python3 mlm_data_preprocessing.py
aws s3 cp ../data/processed s3://data-classification-system/processed/ --recursive
aws s3 cp experiment.log s3://data-classification-system/logs/
#sudo shutdown -h now

#####################################################
#####################################################

#####################################################
#####################################################
# EXPERIMENTS MLM Training:::
mkdir domain_adaption
mkdir models/mlm/
python3 -m venv env_mlm_training
source env_mlm_training/bin/activate
python3 -m pip install torch numpy pandas transformers datasets unidecode
aws s3 cp s3://data-classification-system/src/mlm_train.py .
aws s3 cp s3://data-classification-system/processed/ . --recursive 
python3 mlm_train.py

#!/bin/bash
#chmod 777 ./experiments.sh
#nohup ./experiments.sh &> experiment.log &
source ../env_mlm_training/bin/activate
python3 mlm_train.py
aws s3 cp ../models/mlm/ s3://data-classification-system/models/mlm/ --recursive
aws s3 cp experiment.log s3://data-classification-system/logs/
#sudo shutdown -h now

#####################################################
#####################################################
# EXPERIMENTS MLM Training:::
#!/bin/bash
#chmod 777 ./experiments_v2.sh
#nohup ./experiments_v2.sh &> experiment_mlm.log &
source ~/domain_adaptation/env_mlm_training/bin/activate
echo "Running MLM training..."
python3 mlm_train.py
aws s3 cp ../models/mlm/ s3://data-classification-system/models/mlm/ --recursive
aws s3 cp experiment_mlm.log s3://data-classification-system/logs/
sudo shutdown -h now

#####################################################
#####################################################
# EXPERIMENTS Fine-tuning:::
#!/bin/bash
#chmod 777 ./experiments_fine_tuning.sh
#nohup ./experiments_fine_tuning.sh &> experiments_fine_tuning.log &
source ~/domain_adaptation/env_mlm_training/bin/activate
echo "Running fine tuning..."
python3 fine_tuning.py
aws s3 cp ../models/fine-tuned/ s3://data-classification-system/models/fine-tuned/ --recursive
aws s3 cp experiments_fine_tuning.log s3://data-classification-system/logs/
python3 generate_submissions.py
sudo shutdown -h now

#####################################################
#####################################################

source ../../env_exist2021/bin/activate
exp1=roberta_multitask
echo "${exp1}"
python3 train.py --basenet roberta --task multitask --model_path ../models/$exp1.pt > ./experiments-logs/experiment_$exp1.log 2>&1
python3 generate_submissions.py --basenet roberta --task multitask --model_path ../models/$exp1.pt --output_path ../submissions/submission_$exp1.tsv > ./experiments-logs/submission_$exp1.log 2>&1

gsutil cp ./experiments-logs/experiment_$exp1.log gs://exist2021/experiments-logs
gsutil cp ./experiments-logs/submission_$exp1.log gs://exist2021/experiments-logs
gsutil cp ../models/$exp1.pt gs://exist2021/models
gsutil cp ../submissions/submission_$exp1.tsv gs://exist2021/submissions

#remove .logs and models
rm ./experiments-logs/experiment_$exp1.log
rm ./experiments-logs/submission_$exp1.log
rm ../models/$exp1.pt



