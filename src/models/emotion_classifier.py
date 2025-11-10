from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class EmotionClassifier:
    """Classify emotions in text using a pretrained transformer model."""

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base") -> None:
        """Initialize tokenizer and model.

        Args:
            model_name: The name of the pretrained emotion classification model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Map label IDs to strings using the model's config
        self.labels = {int(k): v for k, v in self.model.config.id2label.items()}

    def predict(self, text: str):
        """Predict emotion for the given text.

        Args:
            text: Input text string.

        Returns:
            A tuple (label, score) representing the most probable emotion and its probability.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = probs.argmax()
        label = self.labels.get(idx, str(idx))
        return label, float(probs[idx])
