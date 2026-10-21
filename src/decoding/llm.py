"""Provide text encoding, decoding, and tokenization for the language model."""
try:
    import re
    import sys
    from typing import Optional
    from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
    from src.llm_sdk import Small_LLM_Model
except KeyboardInterrupt as e:
    print(e)
    sys.exit(0)


def _bytes_to_unicode() -> dict[int, str]:
    """Create a mapping from byte values to Unicode characters.

    Returns:
        A dictionary mapping each byte value to its Unicode representation.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Llm(BaseModel):
    """Handle vocabulary-based text and token conversions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    vocabs: dict[str, int]
    vocabs_str: list[str]
    model_name: Optional[str] = Field(default="Qwen/Qwen3-0.6B")

    _byte_encoder: dict[int, str] = PrivateAttr(
        default_factory=_bytes_to_unicode)
    _byte_decoder: dict[str, int] = PrivateAttr(default_factory=dict)
    _id_to_token: dict[int, str] = PrivateAttr(default_factory=dict)
    _max_token_len: int = PrivateAttr(default=1)
    _special_tokens: set[str] = PrivateAttr(default_factory=set)
    _special_split: Optional[re.Pattern] = PrivateAttr(default=None)
    _pretokenize: Optional[re.Pattern] = PrivateAttr(default=None)

    def model_post_init(self, __context: str) -> None:
        """Initialize tokenization and decoding lookup structures.

        Compile the pre-tokenization pattern and prepare the byte decoder,
        token ID mapping, maximum token length, and special-token matcher.
        """
        self._pretokenize = re.compile(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
            r"|(?:(?![\r\n])[\W_])?[^\W\d_]+"
            r"|\d"
            r"| ?(?:[^\s\w]|_)+[\r\n]*"
            r"|\s*[\r\n]+"
            r"|\s+(?!\S)"
            r"|\s+"
        )
        self._byte_decoder = {c: b for b, c in self._byte_encoder.items()}
        self._id_to_token = {i: t for t, i in self.vocabs.items()}
        self._max_token_len = max(len(t) for t in self.vocabs)

        self._special_tokens = {
            t for t in self.vocabs if re.fullmatch(r"<\|[^|>]+\|>", t)
        }
        if self._special_tokens:
            ordered = sorted(self._special_tokens, key=len, reverse=True)
            self._special_split = re.compile(
                "(" + "|".join(map(re.escape, ordered)) + ")"
            )

    def llm_encode(self, text: str) -> list[int]:
        """Convert text into a sequence of token IDs.

        Split special tokens from ordinary text, pre-tokenize the text,
        map its UTF-8 bytes to Unicode characters, and look up vocabulary IDs.

        Args:
            text: Text to encode.

        Returns:
            A list of token IDs representing the input text.
        """
        ids: list[int] = []
        for part in self._split_special(text):
            if part in self._special_tokens:
                ids.append(self.vocabs[part])
                continue
            if self._pretokenize is not None:
                for chunk in self._pretokenize.findall(part):
                    mapped = "".join(
                        self._byte_encoder[b] for b in chunk.encode("utf-8")
                    )
                    ids.extend(
                        self.vocabs[t] for t in self.llm_tokenize(mapped))
        return ids

    def llm_decode(self, tokens: "list[int] | int") -> str:
        """Convert token IDs back into text.

        Reconstruct the byte sequence from vocabulary tokens and decode it
        as UTF-8.

        Args:
            tokens: A token ID or a list of token IDs.

        Returns:
            The decoded text, replacing invalid UTF-8 sequences when necessary.
        """
        if not hasattr(tokens, "__iter__"):
            tokens = [tokens]
        mapped = "".join(self._id_to_token[int(t)] for t in tokens)

        out = bytearray()
        for c in mapped:
            b = self._byte_decoder.get(c)
            if b is None:
                out += c.encode("utf-8")
            else:
                out.append(b)
        return out.decode("utf-8", errors="replace")

    def llm_tokenize(self, text: str) -> list[str]:
        """Tokenize text using greedy longest-vocabulary matching.

        Args:
            text: A byte-level string to tokenize.

        Returns:
            A list of vocabulary token strings.

        Raises:
            ValueError: If no vocabulary token matches the remaining text.
        """
        result: list[str] = []
        pos, n = 0, len(text)
        while pos < n:
            end = min(n, pos + self._max_token_len)
            while end > pos and text[pos:end] not in self.vocabs:
                end -= 1
            if end == pos:
                raise ValueError(f"Cannot tokenize: {text[pos:]!r}")
            result.append(text[pos:end])
            pos = end
        return result

    def _split_special(self, text: str) -> list[str]:
        """Split text around recognized special tokens.

        Args:
            text: Input text that may contain special tokens.

        Returns:
            A list of text segments and special tokens, preserving their order.
        """
        if self._special_split is None:
            return [text]
        return [p for p in self._special_split.split(text) if p]
