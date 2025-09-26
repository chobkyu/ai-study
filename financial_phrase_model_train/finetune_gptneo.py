import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import transformers

print(transformers.__version__)

def finetune_model():
    # 1. 텍스트 파일 읽기
    file_paths = [
        "Sentences_50Agree.txt",
        "Sentences_66Agree.txt",
        "Sentences_75Agree.txt",
        "Sentences_allAgree.txt"
    ]
    all_sentences = []
    for path in file_paths:
        with open(path, "r", encoding="latin1") as f:
            all_sentences.extend([line.strip() for line in f if line.strip()])

    # 2. 데이터프레임 생성
    df = pd.DataFrame({"text": all_sentences, "label": [0]*len(all_sentences)})

    # 3. Dataset 변환 및 train/test split
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1)

    # 4. 토크나이저 불러오기
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. 토크나이징 + labels 추가
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=64,
        )
        tokenized["labels"] = tokenized["input_ids"]  # 여기 중요
        return tokenized

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # 6. 모델 불러오기
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 7. 학습 설정
    training_args = TrainingArguments(
        output_dir="./gptneo-finetuned",
        # evaluation_strategy="epoch",  # 호환 문제로 주석
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    # 8. Trainer 정의
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # 9. 학습 시작
    trainer.train()
    return "./gptneo-finetuned", tokenizer

if __name__ == "__main__":
    finetune_model()
