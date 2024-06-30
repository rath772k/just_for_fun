from tokenizers.base_tokenizer import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    def __init__(self, char_to_token, token_to_char, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_to_token = char_to_token
        self.token_to_char = token_to_char
        self.vocab_size = len(self.char_to_token)

    def encode(self, text):
        return [self.char_to_token[char] for char in text]

    def decode(self, tokens):
        return "".join(self.token_to_char[token] for token in tokens)

    @classmethod
    def build_tokenizer(cls, filepath):
        with open(filepath, "r") as f:
            data = f.read()
        vocab = set(data)
        char_to_token = {char: idx for idx, char in enumerate(vocab)}
        token_to_char = {idx: char for char, idx in char_to_token.items()}
        return cls(char_to_token, token_to_char)
