import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "StanfordAIMI/RadBERT"
MAX_SEQ_LEN = 192


class RadBERTEncoder:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name).to(self.device)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

    def encode(self, texts, max_len=MAX_SEQ_LEN):
        if isinstance(texts, str):
            texts = [texts]
        encoding = self.tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
