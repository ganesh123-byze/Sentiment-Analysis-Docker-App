import joblib
from src.data_preprocessing import TextPreprocessor


class SentimentPredictor:
    def __init__(self, model_path="models/sentiment_model.pkl"):
        self.model = joblib.load(model_path)
        self.preprocessor = TextPreprocessor()

    def predict(self, text: str):
        # Clean input text
        cleaned_text = self.preprocessor.clean_text(text)

        # Predict
        prediction = self.model.predict([cleaned_text])

        return prediction[0]


# Test block (only runs if executed directly)
if __name__ == "__main__":
    predictor = SentimentPredictor()

    sample_text = "I absolutely love this product!"
    result = predictor.predict(sample_text)

    print("Input:", sample_text)
    print("Predicted Sentiment:", result)
