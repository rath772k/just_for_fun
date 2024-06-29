from tokenizers.base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab, merges, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        self.merges = merges
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            replacement_pair = min([(cur, nxt) for cur, nxt in zip(tokens, tokens[1:])], key=lambda pair: self.merges.get(pair, 1e9))
            if replacement_pair not in self.merges:
                break
            replacement_idx = self.merges[replacement_pair]
            tokens = self.merge(tokens, replacement_pair, replacement_idx)
        return tokens

    def decode(self, tokens):
        byte_string = b"".join(self.vocab[token] for token in tokens)
        return byte_string.decode("utf-8", errors="replace")
    
    @classmethod
    def build_tokenizer(cls, filepath, target_vocab_size):
        with open(filepath, "r") as f:
            data = f.read()
        unicode_data = list(data.encode("utf-8"))
        vocab = {idx: bytes([idx]) for idx in range(256)}
        merges = {}

        for _ in range(target_vocab_size - 256):
            if len(unicode_data) == 1:
                break
            freq = cls.get_freq(unicode_data)
            merge_pair = max(freq, key=freq.get)
            
            new_idx = len(vocab)
            merges[merge_pair] = new_idx
            
            byte_pair = vocab[merge_pair[0]] + vocab[merge_pair[1]]
            vocab[new_idx] = byte_pair

            unicode_data = cls.merge(unicode_data, merge_pair, new_idx)
        
        return cls(vocab, merges)
    
    @staticmethod
    def merge(tokens, pair, idx):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    @staticmethod
    def get_freq(data):
        from collections import defaultdict
        freq = defaultdict(int)
        for cur, nxt in zip(data, data[1:]):
            freq[(cur, nxt)] += 1
        return freq
