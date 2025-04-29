import os
import re
import io
import pickle
import pandas as pd
import numpy as np
import fasttext
import spacy
from pathlib import Path
from docx import Document
import PyPDF2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from rapidfuzz import process, fuzz
from metaphone import doublemetaphone
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

# ======== КОНФИГУРАЦИЯ ========
SVM_MODEL_PATH = Path("svm_sentiment.pkl")
FASTTEXT_MODEL_PATH = Path("cc.uz.300.bin")
EXCEL_FILEPATH = Path("sentences.xlsx")
SHEET_NAME = "Sheet1"
STOP_WORDS = {"bu", "va", "lekin", "uchun", "bilan", "da", "dan",
              "bolib", "edi", "ega", "ham", "masalan", "ammo", "biroq"}
FUZZY_THRESHOLD = 70
N_ESTIMATORS = 100
CALIBRATION_CV = 'prefit'

class SentimentAnalyzer:
    def __init__(self,
                 pos_keywords: dict = None,
                 neg_keywords: dict = None,
                 n_estimators: int = N_ESTIMATORS):
        self.nlp = spacy.blank("xx")
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        if not FASTTEXT_MODEL_PATH.exists():
            raise FileNotFoundError(f"fastText модель не найдена: {FASTTEXT_MODEL_PATH}")
        self.ft_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))

        self.positive_keywords = pos_keywords or {}
        self.negative_keywords = neg_keywords or {}
        self.keyword_set = set(self.positive_keywords) | set(self.negative_keywords)

        self.n_estimators = n_estimators
        if SVM_MODEL_PATH.exists():
            with open(SVM_MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model {SVM_MODEL_PATH} dan yuklandi.")
        else:
            data = self._load_data(EXCEL_FILEPATH, SHEET_NAME)
            self._train_and_calibrate(data)

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'(.)\1{2,}', r'\1', text)  # Нормализация удлиненных символов
        return re.sub(r"[^\w\s]", "", text)

    def _remove_stopwords(self, text: str) -> str:
        return " ".join(w for w in text.split() if w not in STOP_WORDS)

    def _clean(self, text: str) -> str:
        return self._remove_stopwords(self._preprocess(text))

    def _load_data(self, filepath: Path, sheet_name: str):
        df = pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")
        df = df.dropna(subset=["Izoh", "Baho"]).astype(str)
        return list(zip(df["Izoh"].tolist(), df["Baho"].str.lower().tolist()))

    def _get_embedding(self, text: str) -> np.ndarray:
        cleaned = self._clean(text)
        vecs, weights = [], []
        for token in cleaned.split():
            canon = self._map_to_keyword(token)
            weight = self.positive_keywords.get(canon, self.negative_keywords.get(canon, 1.0))
            if canon in self.ft_model:
                vecs.append(self.ft_model.get_word_vector(canon) * weight)
                weights.append(weight)
        if not vecs:
            return np.zeros(self.ft_model.get_dimension())
        return np.sum(vecs, axis=0) / sum(weights)

    def _train_and_calibrate(self, data):
        X = np.vstack([self._get_embedding(t) for t, _ in data])
        y = [lbl for _, lbl in data]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        base_clf = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbose=-1
        )
        
        base_clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric='logloss'
        )
        
        train_accs, test_accs, epochs = [], [], []
        step = max(1, self.n_estimators // 10)
        for e in range(step, self.n_estimators + 1, step):
            temp_clf = LGBMClassifier(
                n_estimators=e,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
                n_jobs=1,
                verbose=-1
            )
            temp_clf.fit(X_train, y_train)
            ta = accuracy_score(y_train, temp_clf.predict(X_train))
            te = accuracy_score(y_test, temp_clf.predict(X_test))
            epochs.append(e)
            train_accs.append(ta)
            test_accs.append(te)
            print(f"Boosting Round {e}/{self.n_estimators} — train_acc: {ta:.3f}, test_acc: {te:.3f}")
            print(classification_report(y_test, temp_clf.predict(X_test)))
        
        calib = CalibratedClassifierCV(estimator=base_clf, cv=CALIBRATION_CV)
        calib.fit(X_train, y_train)
        self.model = calib
        
        with open(SVM_MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        
        plt.figure(figsize=(8,5))
        plt.plot(epochs, train_accs, marker='o', label='Train Acc')
        plt.plot(epochs, test_accs, marker='s', label='Test Acc')
        plt.title('Accuracy per Boosting Round')
        plt.xlabel('Boosting Round'); plt.ylabel('Accuracy')
        plt.xticks(epochs); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig('accuracy_per_round.png', dpi=150)
        print("График сохранён: accuracy_per_round.png")

    def _map_to_keyword(self, token: str) -> str:
        if self.keyword_set:
            best = process.extractOne(token, self.keyword_set, scorer=fuzz.WRatio)
            if best and best[1] >= FUZZY_THRESHOLD:
                return best[0]
        
        ph = doublemetaphone(token)[0]
        if self.keyword_set:
            for kw in self.keyword_set:
                if doublemetaphone(kw)[0] == ph:
                    return kw
        
        norm = token.replace('x', 'h')
        if self.keyword_set:
            best = process.extractOne(norm, self.keyword_set, scorer=fuzz.WRatio)
            if best and best[1] >= FUZZY_THRESHOLD:
                return best[0]
        
        return norm if norm in self.ft_model else token

    def analyze_text(self, text: str) -> str:
        parts = re.split(r"\blekin\b|\bammo\b|\besa\b|\bbiroq\b|[-;,]", text.lower())
        probs = {}
        for part in parts:
            part = part.strip()
            if part:
                emb = self._get_embedding(part)
                if emb.any():
                    preds = self.model.predict_proba([emb])[0]
                    probs = dict(zip(self.model.classes_, preds))
        return max(probs, key=probs.get) if probs else 'neutral'

    def analyze_sentences(self, text: str) -> dict:
        text = text.lower()
        doc = self.nlp(text)
        cnt = {c: 0 for c in self.model.classes_}
        for sent in doc.sents:
            cl = self.analyze_text(sent.text)
            cnt[cl] += 1
        tot = sum(cnt.values())
        return {**cnt,
                'positive_percentage': round(cnt.get('positive', 0) / tot * 100, 2) if tot else 0,
                'total': tot}

    def read_file(self, file_path: str | Path) -> str:
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        ext = file_path.suffix.lower()
        if ext == '.txt':
            return file_path.read_text(encoding='utf-8')
        if ext == '.docx':
            return '\n'.join([p.text for p in Document(file_path).paragraphs])
        if ext == '.pdf':
            t = ''
            for p in PyPDF2.PdfReader(str(file_path)).pages:
                t += p.extract_text() or ''
            return t
        raise ValueError(f"Неподдерживаемый формат {ext}")

if __name__ == '__main__':
    pos_keywords = {}
    neg_keywords = {}
    analyzer = SentimentAnalyzer(pos_keywords=pos_keywords, neg_keywords=neg_keywords)
    analyzer.run_interactive()