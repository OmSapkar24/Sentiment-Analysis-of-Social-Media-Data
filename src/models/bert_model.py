from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class BertSentimentModel:
    """Sentiment analysis using a transformer model (RoBERTa/XLM)."""
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment") -> None:
        """Initialize tokenizer and model.

        Args:
            model_name: Name of the pretrained model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # labels for the CardiffNLP sentiment model: 0=negative, 1=neutral, 2=positive
        self.labels = ["negative", "neutral", "positive"]

    def predict(self, text: str):
        """Predict sentiment for a given text.

        Args:
            text: Input text string.

        Returns:
            Tuple of (label, score) where label is one of negative/neutral/positive and score is probability.
        """
        # Tokenize and truncate to model max length
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = probs.argmax()
        return self.labels[idx], float(probs[idx])
