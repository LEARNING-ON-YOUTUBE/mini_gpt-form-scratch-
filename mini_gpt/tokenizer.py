class CharacterTokenizer:
    def __init__(self, characters: list[str]) -> None:
        if not characters or len(set(characters)) != len(characters):
            raise ValueError("characters must be a non-empty sequence of unique characters")
        self.characters = characters
        self.stoi = {character: index for index, character in enumerate(characters)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[character] for character in text]
        except KeyError as error:
            raise ValueError(f"Character {error.args[0]!r} is not in the vocabulary") from error

    def decode(self, token_ids: list[int]) -> str:
        try:
            return "".join(self.characters[index] for index in token_ids)
        except IndexError as error:
            raise ValueError("Token ID is outside the vocabulary") from error


Tokenizer = CharacterTokenizer
SimpleTokenizer = CharacterTokenizer