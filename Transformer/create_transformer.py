#import torch
#import numpy as np
#import csv
#import pandas as pd
#import pickle
#import matplotlib.pyplot as plt
#import json
from datasets import Dataset
from json import loads, dumps
'''
df_hex_bdw = pd.read_csv("/home/delluser/Documents/ithemal/Ithemal/bdw_uica.csv", usecols = ['hex'], low_memory = False)
df_asm_bdw = pd.read_csv("/home/delluser/Documents/ithemal/Ithemal/bdw_uica.csv", usecols = ['asm'], low_memory = False)
hex_list_bdw1 = df_hex_bdw.values.tolist()
asm_list_bdw1 = df_asm_bdw.values.tolist()

contents1 = [] #hex
contents2 = [] #score
contents4 = [] #asm

with open("/home/delluser/Documents/ithemal/Ithemal/throughput2.csv", 'r') as file2:
  csvreader = csv.reader(file2)
  for row in csvreader:
    contents2.append(float(row[0]))
  
for i in range (len(hex_list_bdw1)):
	contents1.append(hex_list_bdw1[i])
	contents4.append(asm_list_bdw1[i])
    
lista = []
for i in range (len(hex_list_bdw1)):
	l = ("id", "text", "uuid", "score")
	x = (contents1[i][0], str(contents4[i][0]), contents1[i][0], float(contents2[i]))		
	lista_dict = dict(zip(l, x))
	#json_object = json.dumps(lista_dict, indent = 4) 
	lista.append(lista_dict)

#print(lista[0])
#print(lista[1])

df = pd.DataFrame (lista)
#print(df.loc[[0]])
#print(df.loc[[1]])

train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
#print(train)
#print(validate)
#print(test)

#train.to_json(r'/home/delluser/Documents/ithemal/Ithemal/train.json',orient="records")
#validate.to_json(r'/home/delluser/Documents/ithemal/Ithemal/validate.json',orient="records")
#test.to_json(r'/home/delluser/Documents/ithemal/Ithemal/test.json',orient="records")
'''
raw_train_ds = Dataset.from_json("/various/diplom/taivalis/ithemal/transformer/train.json")
raw_val_ds = Dataset.from_json("/various/diplom/taivalis/ithemal/transformer/validate.json")
raw_test_ds = Dataset.from_json("/various/diplom/taivalis/ithemal/transformer/test.json")

# This line prints the description of train_ds
raw_train_ds, raw_val_ds, raw_test_ds

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BigBirdPegasusModel, DistilBertModel
#GPTBigCodeForCasualLM, DataCollatorWithPadding, DistilBertModel
#, AlbertModel, ConvBertModel, AutoModelForSeq2SeqLM, GPTBigCodeForCasualLM
from torch.utils.data import DataLoader

BASE_MODEL = "distilbert-base-uncased"
#"google/bigbird-pegasus-large-arxiv"
#"bigcode/gpt_bigcode-santacoder"#"vinai/bertweet-base" #"distilbert-base-uncased"#"YituTech/conv-bert-base" #"albert-base-v2"#"distilbert-base-uncased"#"camembert-base"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 100 #20


#tokenizer = ElectraTokenizer.from_pretrained(BASE_MODEL)
#model = BigBirdPegasusModel.from_pretrained(BASE_MODEL)
#tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
#retriever = RagRetriever.from_pretrained(BASE_MODEL)
#model = FlaxBartForSequenceClassification.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
ds = {"train": raw_train_ds, "validation": raw_val_ds, "test": raw_test_ds}


def preprocess_function(examples):
    label = examples["score"] 
    examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    examples["label"] = float(label)
    return examples

for split in ds:
    ds[split] = ds[split].map(preprocess_function, remove_columns=["id", "text", "uuid", "score"])


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
    
    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}
    
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="/various/diplom/taivalis/ithemal/transformer",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    do_train = True,
    do_eval = True,
    save_total_limit=2,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    weight_decay=0.01,
)
 
 
 
 
 
import torch

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
        
        
        
trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics_for_regression,
)
trainer.train()
'''
print("train_evaluation")
trainer.eval_dataset = ds["train"]
metrics = trainer.evaluate()
print(metrics)

print("validate_evaluation")
trainer.eval_dataset = ds["validation"]
metrics = trainer.evaluate()
print(metrics)
'''
print("test_evaluation")
predictions = trainer.predict(ds["test"])
arr = predictions.predictions
lab = predictions.label_ids
file = open('/various/diplom/taivalis/ithemal/transformer/save_transf.txt', 'w')
for i in range (len (arr)):
    file.write(str(arr[i]))
    file.write("\n")
    file.write(str(lab[i]))
    file.write("\n")
file.close()
