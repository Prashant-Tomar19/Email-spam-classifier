import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Load dataset
df = pd.read_csv("spam.csv")   # adjust if your file has a different name

# 2. Split data
X = df["message"]
y = df["label"].map({"ham": 0, "spam": 1})   # convert to 0/1

# 3. Create pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

# 4. Train
model.fit(X, y)

# 5. Save pipeline
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
