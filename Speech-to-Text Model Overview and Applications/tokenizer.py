
class CharTokenizer:
    def __init__(self, chars):
        self.char_to_idx = {char: i + 1 for i, char in enumerate(chars)} # 0 for blank
        self.idx_to_char = {i + 1: char for i, char in enumerate(chars)}
        self.idx_to_char[0] = ">blank<"
        self.vocab_size = len(chars) + 1

    def encode(self, text):
        return [self.char_to_idx.get(char, 0) for char in text.lower()]

    def decode(self, tokens):
        return "".join([self.idx_to_char.get(token, ">") for token in tokens])

if __name__ == "__main__":
    # Example usage
    chars = "abcdefghijklmnopqrstuvwxyz \n'"
    tokenizer = CharTokenizer(chars)
    text = "hello world"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.vocab_size}")


