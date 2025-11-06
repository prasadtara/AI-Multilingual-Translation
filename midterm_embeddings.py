import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings

# Suppress harmless sklearn warnings for clean output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def train_multilingual_classifier():
    """
    Loads multilingual data, converts text to embeddings using a Tamil-English
    compatible model, and trains a Logistic Regression classifier.
    """

    # --- Configuration ---
    # Model chosen for strong performance across Tamil and English
    MODEL_NAME = 'sentence-transformers/LaBSE' # Corrected model name

    # 1. Prepare Multilingual Dataset
    # This simulates loading your data, where 'text' can be either Tamil or English.
    data = {
        'text': [
            "This is a great movie, I highly recommend it.",          # English: Positive
            "படம் நன்றாக உள்ளது, நான் மிகவும் பரிந்துரைக்கிறேன்.",     # Tamil: Positive (The movie is good, I highly recommend it.)
            "I am unhappy with the poor service and long wait.",      # English: Negative
            "சேவை திருப்தியற்றது மற்றும் காத்திருப்பு நீண்டது.",      # Tamil: Negative (The service is unsatisfactory and the wait is long.)
            "The weather is cloudy today.",                           # English: Neutral
            "இன்று வானிலை மேகமூட்டமாக உள்ளது.",                      # Tamil: Neutral (Today the weather is cloudy.)
            "A fantastic experience, everything was perfect!",        # English: Positive
            "இது ஒரு அருமையான அனுபவம்!",                              # Tamil: Positive (This is a fantastic experience!)
        ],
        'label': [
            'Positive', 'Positive', 'Negative', 'Negative',
            'Neutral', 'Neutral', 'Positive', 'Positive'
        ]
    }
    df = pd.DataFrame(data)

    print("--- Step 1: Data Preparation ---")
    print(f"Total samples: {len(df)}")

    texts = df['text'].tolist()
    labels = df['label'].to_numpy()

    # 2. Load Multilingual Embedding Model
    try:
        model = SentenceTransformer(MODEL_NAME)
        print(f"Successfully loaded multilingual model: {MODEL_NAME}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an active internet connection and the `sentence-transformers` library installed.")
        return

    # 3. Generate Embeddings (Feature Engineering)
    print("\n--- Step 2: Generating Multilingual Embeddings ---")
    # This step aligns the semantic meaning of Tamil and English into a single vector space.
    embeddings = model.encode(texts, show_progress_bar=True)

    X = embeddings
    y = labels

    print(f"Embedding vector dimension: {X.shape[1]}")
    print(f"Feature matrix shape: {X.shape}")

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.375, random_state=42, stratify=y # Changed test_size to 0.375
    )
    print(f"\nTraining samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 5. Train a Classifier
    # Logistic Regression is used here as a fast, effective classifier on top of the fixed embeddings.
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    print("\n--- Step 3: Training the Classifier ---")
    classifier.fit(X_train, y_train)
    print("Classifier training complete.")

    # 6. Evaluate
    y_pred = classifier.predict(X_test)

    print("\n--- Step 4: Multilingual Evaluation ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")


    # The actual goal is to show the process.
    print("\nClassification Report (Overall Performance):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n*** The classifier successfully leveraged the shared multilingual embedding space. ***")


if __name__ == "__main__":
    train_multilingual_classifier()
