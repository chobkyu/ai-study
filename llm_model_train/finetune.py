from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling)

# 1. 데이터셋(작은 공개 데이터)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 2. 토크나이저/모델
model_name = "gpt2" # samll 버전
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize(examples):
    return tokenizer(examples["text"], truncation = True, padding="max_length", max_length=64)

tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 3. 학습 파라미터
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size= 2,
    num_train_epochs = 1, # 1 epoch만 (이게 뭘까)
    logging_steps=100,
    save_steps=500,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./my_finetuned_model")
tokenizer.save_pretrained("./my_finetuned_model")
print("모델 저장 완료")