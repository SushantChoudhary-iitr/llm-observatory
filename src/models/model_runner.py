from transformers import AutoTokenizer, AutoModel
import torch


class ModelRunner:

    #initializing the model, attributes
    def __init__(self,
                 model_name="distilbert-base-uncased",
                 device=None):

        self.model_name = model_name

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = None
        self.model = None


    def load_model(self):

        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModel.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded.")


    def infer(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs