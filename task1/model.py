# model.py
from transformers import RobertaForSequenceClassification

class RobertaModel:
    def __init__(self, num_labels=10):
        self.num_labels = num_labels
        self.model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base', num_labels=self.num_labels
        )

    def get_model(self):
        return self.model
