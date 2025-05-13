import re
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset

# sozlamalar
MODEL_NAME = "DGurgurov/xlm-r_uzbek_sentiment"
OUTPUT_DIR = Path("./fine_tuned_xlmr_uzbek")
EXCEL_FILE = Path("sentences.xlsx")
SHEET = "Sheet1"
# klasslar
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# normalizatsiya
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s/%]", "", text)  # удаляем всё, кроме букв/цифр, пробелов, %, /
    text = re.sub(r"\s+", " ", text)
    return text

# data baza
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = [normalize_text(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# trein
class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True
        ).to(DEVICE)

        if OUTPUT_DIR.exists():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
                self.model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(DEVICE)
                print("Loaded fine-tuned model and tokenizer from disk.")
            except Exception:
                print("Incomplete OUTPUT_DIR; retraining.")
                self._fine_tune()
        else:
            self._fine_tune()

    def _load_data(self):
        df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET, engine='openpyxl')
        df = df.dropna(subset=["Izoh", "Baho"]).copy()
        df['text'] = df['Izoh'].astype(str)
        # Обработка трех классов, пропускаем неизвестные
        df['label'] = df['Baho'].str.lower().map(LABEL2ID)
        df = df.dropna(subset=['label'])
        texts = df['text'].tolist()
        labels = df['label'].astype(int).tolist()
        return train_test_split(
            texts, labels,
            test_size=0.1,
            random_state=SEED,
            stratify=labels
        )

    def _fine_tune(self):
        train_texts, val_texts, train_labels, val_labels = self._load_data()
        train_ds = SentimentDataset(train_texts, train_labels, self.tokenizer)
        val_ds = SentimentDataset(val_texts, val_labels, self.tokenizer)
        data_collator = DataCollatorWithPadding(self.tokenizer)

        # class weights 
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(list(LABEL2ID.values())),
            y=train_labels
        )
        class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            warmup_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=50,
            seed=SEED,
        )

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids
            accuracy = (preds == labels).mean()
            return {"accuracy": accuracy}

        class WeighedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
                labels = inputs.pop('labels')
                outputs = model(**inputs)
                logits = outputs.logits
                loss = torch.nn.CrossEntropyLoss(weight=class_weights)(logits, labels)
                return (loss, outputs) if return_outputs else loss

        trainer = WeighedTrainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        trainer.train()
        trainer.save_model(OUTPUT_DIR)
        self.tokenizer.save_pretrained(OUTPUT_DIR)
        print("Fine-tuning completed.")

    def read_file(self, file_path: str | Path) -> str:
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        ext = file_path.suffix.lower()
        if ext == '.txt':
            return file_path.read_text(encoding='utf-8')
        if ext == '.docx':
            from docx import Document
            return '\n'.join([p.text for p in Document(file_path).paragraphs])
        if ext == '.pdf':
            import PyPDF2
            t = ''
            for p in PyPDF2.PdfReader(str(file_path)).pages:
                t += p.extract_text() or ''
            return t
        raise ValueError(f"Неподдерживаемый формат файла {ext}")

    def analyze_sentences(self, text: str) -> dict:
        sentences = re.split(r'[.!?]\s*', text)
        cnt = {label: 0 for label in LABEL2ID}
        for sent in sentences:
            sent = sent.strip()
            if sent:
                label = self.analyze_text(sent)
                cnt[label] += 1
        tot = sum(cnt.values())
        return {**cnt,
                'positive_percentage': round(cnt.get('positive', 0) / tot * 100, 2) if tot else 0,
                'total': tot}   
    

    def analyze_text(self, text: str) -> str:
        print(text)
        # rokenlash
        normalized_text = normalize_text(text)
        print(normalized_text)
        inputs = self.tokenizer(normalized_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # tokenlarni chiqarish
        tokens = self.tokenizer.tokenize(normalized_text)
        print("\nTokens:", tokens)

        
        # classes
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            softmax = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
            class_idx = softmax.argmax()
            label = ID2LABEL[class_idx]
            print(f"\nPredicted: {label}, Probabilities: {softmax}")
            return label

    def predict(self, texts):
        single = isinstance(texts, str)
        inputs = self.tokenizer(
            [texts] if single else texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        ).to(DEVICE)
        outs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        labels = [ID2LABEL[int(p)] for p in preds]
        return labels[0] if single else labels

if __name__ == '__main__':
    SentimentAnalyzer()