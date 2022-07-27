#%% 
import os
import datasets
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
#%%
import torch, gc
torch.cuda.empty_cache()
gc.collect()
#%% inputs
sample = 10
TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 5e-4 
LR_WARMUP_STEPS = 10000
WEIGHT_DECAY = 0.01
SEED_TRAIN = 0
data_path_save = "data/processed"
language = "legacy"
MODEL = "nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large"
MODEL_tokenizer = "xlm-roberta-base"
model_path_save = "models/mlm/xmlr_mlm/"
model_path_save = os.path.join("../",model_path_save)
model_checkpoint = "model_checkpoints/xlm-r-sexism"
model_checkpoint=os.path.join("../",model_checkpoint)
EPOCHS = 1
#%% 
print("-------------- MLM Preprocessing INPUTS: --------------")
print("Data training path: " , data_path_save)
print("Model: ", MODEL)
print("Path model: ", model_path_save)
print("Sample raw data: ", sample)

#%%
print("Loading data...")
if not language:
    data_path_save = "data/processed"
    train_en = datasets.load_from_disk(os.path.join("../",data_path_save, 'en'))
    train_es = datasets.load_from_disk(os.path.join("../",data_path_save, 'es'))
    train_dataset = datasets.DatasetDict({ 
        "train": datasets.concatenate_datasets(
        [train_en['train'], train_es['train']])
    })
elif language=="legacy":
    data_path_save=os.path.join("../",data_path_save)
    train_dataset = datasets.load_from_disk(data_path_save)
else:
    data_path_save=os.path.join("../",data_path_save, language)
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

trainer.train()

#%% 
print("Saving model at ", model_path_save)
trainer.save_model(model_path_save)
print("Processed finished!")
# %%
