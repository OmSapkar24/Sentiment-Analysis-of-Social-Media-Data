class EnsembleSentimentModel:
    """Combine multiple sentiment models using weighted voting."""

    def __init__(self, models):
        """Initialize ensemble with a list of models implementing predict(text)."""
        self.models = models

    def predict(self, text: str):
        """Aggregate predictions from all models.

        Args:
            text: Input text string.

        Returns:
            A tuple (label, confidence) representing the majority sentiment and normalized confidence.
        """
        # Accumulate scores for each label
        label_scores = {}
        for model in self.models:
            label, score = model.predict(text)
            label_scores[label] = label_scores.get(label, 0.0) + score
        # Pick label with highest aggregated score
        if not label_scores:
            return None, 0.0
        final_label = max(label_scores, key=label_scores.get)
        total_score = sum(label_scores.values())
        confidence = label_scores[final_label] / total_score if total_score > 0 else 0.0
        return final_label, confidence
