const fs = require('fs');
const path = require('path');

const targetDir = 'C:\\\\Users\\\\salun\\\\OneDrive - smarttech\\\\Documents\\\\D Drive backup\\\\creatosaurus-intership\\\\cache_backend\\\\listening-service\\\\nlp-service';

// Update train.py
const trainPy = `import os
import sys
import json
from pure_bayes import PureTFIDFNaiveBayes
from preprocessor import clean_text, extract_features_text

def load_dataset(dataset_path: str, max_samples: int = 20000):
    print(f"[Train] Loading dataset from: {dataset_path} (max_samples={max_samples})")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    texts = []
    labels = []

    with open(dataset_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            if len(texts) >= max_samples:
                break
            title = item.get('title', '')
            content = item.get('content', '')
            label = item.get('sentimental', item.get('sentiment', '')).strip()

            if not label or label.lower() not in ['positive', 'negative', 'neutral']:
                continue

            label = label.title()
            feat_text = extract_features_text(title, content)

            if feat_text:
                texts.append(feat_text)
                labels.append(label)

    print(f"[Train] Extracted {len(texts)} valid text samples with sentiment labels.")
    return texts, labels

def train_nlp_model(dataset_path: str, models_dir: str = None, max_samples: int = 20000):
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(__file__), 'models')

    os.makedirs(models_dir, exist_ok=True)
    texts, labels = load_dataset(dataset_path, max_samples=max_samples)

    if len(texts) < 5:
        print("[Train] Error: Not enough data samples to train model.")
        return False

    split_idx = int(len(texts) * 0.8)
    X_train, X_test = texts[:split_idx], texts[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    print("[Train] Fitting Pure Python TF-IDF Naive Bayes Model...")
    model = PureTFIDFNaiveBayes(ngram_range=(1, 2), max_features=8000)
    model.fit(X_train, y_train)

    correct = 0
    for text, true_label in zip(X_test, y_test):
        pred = model.predict(text)
        if pred == true_label:
            correct += 1

    acc = (correct / max(1, len(y_test))) * 100
    print(f"[Train] Test Accuracy: {acc:.2f}% ({correct}/{len(y_test)})")

    save_path = os.path.join(models_dir, 'pure_sentiment_model.json')
    model.save(save_path)

    print(f"[Train] Saved model artifact to: {save_path}")
    return True

if __name__ == '__main__':
    default_dataset = os.path.join(os.path.dirname(__file__), '..', 'temp_data', 'cache1.listenings.json')
    if len(sys.argv) > 1:
        default_dataset = sys.argv[1]
    
    train_nlp_model(default_dataset)
`;
fs.writeFileSync(path.join(targetDir, 'train.py'), trainPy, 'utf8');

console.log('Updated train.py with max_samples limit');
