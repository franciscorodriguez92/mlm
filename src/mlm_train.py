#%% 
import os
import datasets
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
import json
from pathlib import Path
from transformers.trainer_utils import set_seed
set_seed(123)
#%%
import torch, gc
torch.cuda.empty_cache()
gc.collect()
#import os
#os.environ["NCCL_DEBUG"] = "INFO"
#%% inputs
""" sample = 10
TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 5e-4 
LR_WARMUP_STEPS = 10000
WEIGHT_DECAY = 0.01
SEED_TRAIN = 0
data_path_save = "data/processed"
language = None
MODEL = "nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large"
MODEL_tokenizer = "xlm-roberta-base"
model_path_save = "models/mlm/xmlr_mlm/"
model_path_save = os.path.join("../",model_path_save)
model_checkpoint = "model_checkpoints/xlm-r-sexism"
model_checkpoint=os.path.join("../",model_checkpoint)
EPOCHS = 2 """

#%%
config_file = open('config.json')
config = json.load(config_file)

#%% inputs
SEED_TRAIN = 0
sample = config["mlm_training"]["sample"]
data_path_save = config["preprocessing"]["data_processing_save"]
language = config["mlm_training"]["language"]
model_path_save = config["mlm_training"]["MODEL_PATH_SAVE"]
model_checkpoint = config["mlm_training"]["MODEL_CHECKPOINT"]
TRAIN_BATCH_SIZE = config["mlm_training"]["TRAIN_BATCH_SIZE"]
LEARNING_RATE = config["mlm_training"]["LEARNING_RATE"]
LR_WARMUP_STEPS = config["mlm_training"]["LR_WARMUP_STEPS"]
WEIGHT_DECAY = config["mlm_training"]["WEIGHT_DECAY"]
EPOCHS = config["mlm_training"]["EPOCHS"]
MODEL = config["mlm_training"]["MODEL"]
MODEL_tokenizer = config["preprocessing"]["tokenizer"]
Path(model_path_save).mkdir(parents=True, exist_ok=True) 
Path(model_checkpoint).mkdir(parents=True, exist_ok=True) 

#%% 
print("-------------- MLM Preprocessing INPUTS: --------------")
print("Data training path: " , data_path_save)
print("Model: ", MODEL)
print("Path model: ", model_path_save)
print("Sample raw data: ", sample)

#%%
print("Loading data...")
if not language:
    data_path_save = config["preprocessing"]["data_processing_save"]
    train_en = datasets.load_from_disk(os.path.join(data_path_save, 'en'))
    train_es = datasets.load_from_disk(os.path.join(data_path_save, 'es'))
    train_dataset = datasets.DatasetDict({ 
        "train": datasets.concatenate_datasets(
        [train_en['train'], train_es['train']])
    })
elif language=="legacy":
    train_dataset = datasets.load_from_disk(data_path_save)
else:
    data_path_save=os.path.join(data_path_save, language)
    train_dataset = datasets.load_from_disk(data_path_save)

#%%
if sample != False:
    print('Sampling ', sample,' rows from dataset')
    train_dataset = train_dataset["train"].train_test_split(
        train_size=sample, test_size=1,  seed=42
    )

#%% 
model = AutoModelForMaskedLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL_tokenizer)

#%%
#steps_per_epoch = int(len(train_dataset) / TRAIN_BATCH_SIZE)
print("Train starting...")
training_args = TrainingArguments(
    output_dir=model_checkpoint,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    warmup_steps=LR_WARMUP_STEPS,
    save_total_limit=3,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE, 
    save_strategy='epoch',
    metric_for_best_model='loss', 
    seed=SEED_TRAIN,
    remove_unused_columns=False,
    disable_tqdm = False,
    fp16=True,
    logging_steps=len(train_dataset["train"]) // TRAIN_BATCH_SIZE,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    #data_collator=data_collator,
    #data_collator=whole_word_masking_data_collator,
    train_dataset=train_dataset['train'],
    #eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)

#trainer.train(resume_from_checkpoint=True)
trainer.train()

#%% 
print("Saving model at ", model_path_save)
trainer.save_model(model_path_save)
print("Processed finished!")
# %%
