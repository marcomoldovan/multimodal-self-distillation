from transformers import PerceiverTokenizer

class WikipediaCollator:
    def __init__(
        self,
        pretrained_tokenizer: str = "deepmind/language-perceiver",
        train_on_long_form_text: bool = False,
        max_seq_length: int = 512,
        ):
        self.tokenizer = PerceiverTokenizer.from_pretrained(pretrained_tokenizer)
        self.train_on_long_form_text = train_on_long_form_text
        self.max_seq_length = max_seq_length
        
        
    def __call__(self, batch):
        return self.tokenizer(
            batch, 
            padding='longest', 
            max_length=self.max_seq_length, 
            truncation=True, 
            return_tensors='pt'
            )
