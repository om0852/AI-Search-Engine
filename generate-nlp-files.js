const fs = require('fs');
const path = require('path');

const targetDir = 'C:\\\\Users\\\\salun\\\\OneDrive - smarttech\\\\Documents\\\\D Drive backup\\\\creatosaurus-intership\\\\cache_backend\\\\listening-service\\\\nlp-service';
const modelsDir = path.join(targetDir, 'models');

if (!fs.existsSync(targetDir)) fs.mkdirSync(targetDir, { recursive: true });
if (!fs.existsSync(modelsDir)) fs.mkdirSync(modelsDir, { recursive: true });

// 1. preprocessor.py
const preprocessorPy = `import re
import html

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\\S+|www\\.\\S+', ' ', text)
    text = re.sub(r'@\\w+', ' ', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\\?{2,}', '?', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def extract_features_text(title: str = "", content: str = "") -> str:
    cleaned_title = clean_text(title or "")
    cleaned_content = clean_text(content or "")
    if cleaned_title and cleaned_content:
        return f"{cleaned_title}. {cleaned_content}"
    return cleaned_title or cleaned_content or ""
`;
fs.writeFileSync(path.join(targetDir, 'preprocessor.py'), preprocessorPy, 'utf8');

// 2. engine.py
const enginePy = `import os
import re
import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocessor import clean_text, extract_features_text

class NLPEngine:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.models_dir = models_dir
        self.vader = SentimentIntensityAnalyzer()
        self.model = None
        self.vectorizer = None
        self.load_models()

        # Urgency triggers
        self.critical_keywords = {
            'outage', 'down', 'hack', 'hacked', 'scam', 'fraud', 'breach', 'lawsuit',
            'legal', 'scandal', 'bankruptcy', 'crash', 'recalled', 'warning', 'fire',
            'fatal', 'danger', 'stolen', 'exploit', 'emergency'
        }
        self.high_keywords = {
            'issue', 'bug', 'broken', 'error', 'failed', 'failure', 'worst', 'terrible',
            'horrible', 'refund', 'complaint', 'scammed', 'poor', 'slow', 'lost', 'delay'
        }

        # Topic taxonomy
        self.topic_rules = {
            'Finance & Valuation': ['valuation', 'billion', 'million', 'investment', 'funding', 'stock', 'share', 'ipo', 'profit', 'revenue', 'quarter', 'market cap'],
            'Product & Technology': ['feature', 'update', 'app', 'platform', 'tool', 'ai', 'software', 'tech', 'design', 'mvp', 'launch', 'release', 'system'],
            'Marketing & Social': ['campaign', 'brand', 'instagram', 'linkedin', 'twitter', 'youtube', 'facebook', 'influencer', 'ad', 'marketing', 'post', 'views'],
            'Education & Career': ['course', 'training', 'job', 'hiring', 'internship', 'student', 'university', 'career', 'designer', 'manager', 'role'],
            'Corporate & Governance': ['authority', 'government', 'region', 'development', 'official', 'policy', 'council', 'court', 'sector', 'corp']
        }

    def load_models(self):
        model_path = os.path.join(self.models_dir, 'sentiment_model.joblib')
        vectorizer_path = os.path.join(self.models_dir, 'vectorizer.joblib')

        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                print("[NLP Engine] Successfully loaded ML classifier models.")
            except Exception as e:
                print(f"[NLP Engine] Warning loading ML models: {e}")
                self.model = None
                self.vectorizer = None
        else:
            print("[NLP Engine] ML model files not found. Operating with VADER + Rule-based Engine.")

    def detect_emotions(self, text: str, compound_score: float) -> list:
        text_lower = text.lower()
        emotions = []

        joy_score = 0.0
        anger_score = 0.0
        sadness_score = 0.0
        fear_score = 0.0
        surprise_score = 0.0

        if compound_score >= 0.3:
            joy_score = min(1.0, round(compound_score, 2))
        elif compound_score <= -0.3:
            if any(w in text_lower for w in ['hate', 'angry', 'rage', 'worst', 'furious', 'scam', 'useless']):
                anger_score = min(1.0, round(abs(compound_score), 2))
            elif any(w in text_lower for w in ['sad', 'disappointed', 'unfortunate', 'pity', 'missed']):
                sadness_score = min(1.0, round(abs(compound_score), 2))
            elif any(w in text_lower for w in ['warning', 'fear', 'scared', 'risk', 'danger', 'threat']):
                fear_score = min(1.0, round(abs(compound_score), 2))
            else:
                sadness_score = 0.4
                anger_score = 0.4

        if any(w in text_lower for w in ['wow', 'amazing', 'shocking', 'surprise', 'unbelievable', 'hit']):
            surprise_score = 0.65

        emotions.append({"label": "joy", "score": joy_score})
        emotions.append({"label": "anger", "score": anger_score})
        emotions.append({"label": "sadness", "score": sadness_score})
        emotions.append({"label": "fear", "score": fear_score})
        emotions.append({"label": "surprise", "score": surprise_score})
        emotions.append({"label": "neutral", "score": round(1.0 - max(joy_score, anger_score, sadness_score, fear_score, surprise_score), 2)})

        return sorted(emotions, key=lambda x: x['score'], reverse=True)

    def detect_urgency(self, text: str, sentiment_label: str) -> str:
        words = set(re.findall(r'\\w+', text.lower()))
        if words.intersection(self.critical_keywords):
            return 'critical'
        if words.intersection(self.high_keywords) or sentiment_label == 'Negative':
            return 'high'
        if sentiment_label == 'Positive':
            return 'low'
        return 'medium'

    def extract_topics(self, text: str) -> list:
        text_lower = text.lower()
        detected = []
        for topic, keywords in self.topic_rules.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(topic)
        if not detected:
            detected.append('General News & Media')
        return detected

    def extract_keywords(self, text: str, top_n: int = 5) -> list:
        words = re.findall(r'\\b[A-Za-z]{3,}\\b', text)
        stopwords = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'you', 'are', 'was', 'were', 'have', 'has', 'had', 'not', 'but', 'can', 'will', 'just', 'more', 'about', 'out', 'all', 'one'}
        filtered = [w for w in words if w.lower() not in stopwords]
        
        freq = {}
        for w in filtered:
            key = w.title() if len(w) > 3 else w.upper()
            freq[key] = freq.get(key, 0) + 1
            
        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [k for k, v in sorted_kw[:top_n]]

    def analyze(self, title: str = "", content: str = "", source: str = "") -> dict:
        full_text = extract_features_text(title, content)
        if not full_text:
            return {
                "sentimental": "Neutral",
                "sentimentScore": 0.0,
                "sentimentConfidence": 0.5,
                "emotions": [{"label": "neutral", "score": 1.0}],
                "urgency": "low",
                "topics": ["General"],
                "keywords": [],
                "nlpProcessed": True
            }

        # 1. Lexicon Sentiment (VADER)
        vader_res = self.vader.polarity_scores(full_text)
        compound = vader_res['compound']

        # 2. ML Sentiment (if available)
        ml_probs = None
        ml_label = None
        if self.model is not None and self.vectorizer is not None:
            try:
                vec = self.vectorizer.transform([full_text])
                probs = self.model.predict_proba(vec)[0]
                classes = list(self.model.classes_)
                ml_probs = dict(zip(classes, probs))
                ml_label = self.model.predict(vec)[0]
            except Exception as e:
                pass

        # 3. Hybrid Label Fusion
        if compound >= 0.05:
            vader_label = "Positive"
        elif compound <= -0.05:
            vader_label = "Negative"
        else:
            vader_label = "Neutral"

        if ml_label:
            if ml_label == vader_label:
                final_label = ml_label
                confidence = max(0.7, float(ml_probs.get(ml_label, 0.7)))
            else:
                if abs(compound) > 0.4:
                    final_label = vader_label
                    confidence = 0.65
                else:
                    final_label = ml_label
                    confidence = float(ml_probs.get(ml_label, 0.6))
        else:
            final_label = vader_label
            confidence = min(0.95, max(0.5, 0.5 + abs(compound) * 0.45))

        emotions = self.detect_emotions(full_text, compound)
        urgency = self.detect_urgency(full_text, final_label)
        topics = self.extract_topics(full_text)
        keywords = self.extract_keywords(full_text)

        return {
            "sentimental": final_label,
            "sentimentScore": round(float(compound), 3),
            "sentimentConfidence": round(float(confidence), 3),
            "emotions": emotions,
            "urgency": urgency,
            "topics": topics,
            "keywords": keywords,
            "nlpProcessed": True
        }
`;
fs.writeFileSync(path.join(targetDir, 'engine.py'), enginePy, 'utf8');

// 3. train.py
const trainPy = `import os
import sys
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocessor import clean_text, extract_features_text

def load_dataset(dataset_path: str):
    print(f"[Train] Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    texts = []
    labels = []

    with open(dataset_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
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

def train_nlp_model(dataset_path: str, models_dir: str = None):
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(__file__), 'models')

    os.makedirs(models_dir, exist_ok=True)
    texts, labels = load_dataset(dataset_path)

    if len(texts) < 10:
        print("[Train] Error: Not enough data samples to train model.")
        return False

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels)) > 1 else None
    )

    print("[Train] Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        stop_words='english',
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[Train] Training Calibrated Logistic Regression Classifier...")
    base_clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid')
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"[Train] Test Accuracy: {acc * 100:.2f}%")
    print("[Train] Classification Report:")
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(models_dir, 'sentiment_model.joblib')
    vectorizer_path = os.path.join(models_dir, 'vectorizer.joblib')

    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"[Train] Saved model artifacts to:\\n  - {model_path}\\n  - {vectorizer_path}")
    return True

if __name__ == '__main__':
    default_dataset = os.path.join(os.path.dirname(__file__), '..', 'temp_data', 'cache1.listenings.json')
    if len(sys.argv) > 1:
        default_dataset = sys.argv[1]
    
    train_nlp_model(default_dataset)
`;
fs.writeFileSync(path.join(targetDir, 'train.py'), trainPy, 'utf8');

// 4. main.py
const mainPy = `import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from engine import NLPEngine
from train import train_nlp_model

app = FastAPI(
    title="Render Free-Tier Listening NLP Service",
    description="High-performance, lightweight NLP Microservice for Sentiment, Emotion, Urgency, & Topic Analysis",
    version="1.0.0"
)

nlp_engine = NLPEngine()

class ItemInput(BaseModel):
    title: Optional[str] = ""
    content: Optional[str] = ""
    source: Optional[str] = ""

class BatchInput(BaseModel):
    items: List[ItemInput]

class EmotionDetail(BaseModel):
    label: str
    score: float

class NLPResponse(BaseModel):
    sentimental: str
    sentimentScore: float
    sentimentConfidence: float
    emotions: List[EmotionDetail]
    urgency: str
    topics: List[str]
    keywords: List[str]
    nlpProcessed: bool = True

class TrainRequest(BaseModel):
    dataset_path: Optional[str] = None

@app.get("/")
def read_root():
    return {
        "service": "Listening NLP Service",
        "status": "online",
        "ml_model_loaded": nlp_engine.model is not None,
        "render_tier": "free"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ml_model_active": nlp_engine.model is not None
    }

@app.post("/analyze", response_model=NLPResponse)
def analyze_single(item: ItemInput):
    try:
        return nlp_engine.analyze(title=item.title, content=item.content, source=item.source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch", response_model=List[NLPResponse])
def analyze_batch(payload: BatchInput):
    try:
        results = []
        for item in payload.items:
            res = nlp_engine.analyze(title=item.title, content=item.content, source=item.source)
            results.append(res)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train_endpoint(req: TrainRequest, background_tasks: BackgroundTasks):
    dataset = req.dataset_path or os.path.join(os.path.dirname(__file__), '..', 'temp_data', 'cache1.listenings.json')
    if not os.path.exists(dataset):
        raise HTTPException(status_code=400, detail=f"Dataset path {dataset} does not exist.")

    def run_training():
        success = train_nlp_model(dataset)
        if success:
            nlp_engine.load_models()

    background_tasks.add_task(run_training)
    return {"message": "NLP model training triggered in background.", "dataset": dataset}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
`;
fs.writeFileSync(path.join(targetDir, 'main.py'), mainPy, 'utf8');

// 5. Dockerfile
const dockerfile = `FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV PORT=8000
CMD ["python", "main.py"]
`;
fs.writeFileSync(path.join(targetDir, 'Dockerfile'), dockerfile, 'utf8');

// 6. render.yaml
const renderYaml = `services:
  - type: web
    name: listening-nlp-service
    env: python
    buildCommand: pip install -r nlp-service/requirements.txt
    startCommand: python nlp-service/main.py
    plan: free
    region: singapore
    envVars:
      - key: PORT
        value: 10000
      - key: PYTHON_VERSION
        value: 3.11.0
`;
fs.writeFileSync(path.join(targetDir, 'render.yaml'), renderYaml, 'utf8');

// 7. README.md
const readmeMd = `# Render Free-Tier Listening NLP Service

High-Performance, Lightweight NLP Microservice built with FastAPI, Scikit-Learn, and VADER Sentiment Analysis for Brand & Social Media Listening.

## Features
- **Ultra Low Memory Footprint**: Optimized for Render Free Tier (< 100MB RAM, instant boot).
- **Hybrid Sentiment Engine**: ML Classifier + VADER Lexicon + Rule Fusion.
- **Sentiment Scores & Confidence**: Returns fine-grained score (-1.0 to +1.0) and confidence.
- **Emotion Analysis**: Detects joy, anger, sadness, fear, surprise, and neutral.
- **Urgency Classification**: Categorizes post urgency (critical, high, medium, low).
- **Topic & Keyword Extraction**: Auto-detects brand categories & top entities.

## API Endpoints
- \`GET /health\`: Health check for Render load balancers.
- \`POST /analyze\`: Single item text analysis.
- \`POST /analyze-batch\`: Bulk text analysis for high throughput ingestion.
- \`POST /train\`: Retrains the model on updated dataset.

## How to Train Model Locally
\`\`\`bash
cd nlp-service
pip install -r requirements.txt
python train.py
\`\`\`

## How to Run Locally
\`\`\`bash
python main.py
\`\`\`

## Deploying to Render Free Tier
1. Connect your repository to Render.
2. Select **Web Service** or use \`render.yaml\` blueprint.
3. Set Build Command: \`pip install -r nlp-service/requirements.txt\`
4. Set Start Command: \`python nlp-service/main.py\`
5. Copy your Render service URL (e.g. \`https://listening-nlp-service.onrender.com\`) to your Node.js \`.env\` file as \`NLP_SERVICE_URL\`.
`;
fs.writeFileSync(path.join(targetDir, 'README.md'), readmeMd, 'utf8');

console.log('All nlp-service files generated successfully!');
