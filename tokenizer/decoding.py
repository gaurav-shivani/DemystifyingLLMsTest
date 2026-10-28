"""Token decoding."""
from typing import List, Optional
from .vocab import Vocabulary
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


def decode(ids: List[int], vocab: Optional[Vocabulary] = None, skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs back to a string.
    """
    if vocab is None:
        vocab = Vocabulary()

    tokens = [vocab.id_to_token(int(i)) for i in ids]

    if skip_special_tokens:
        special_tokens = {PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN}
        tokens = [
            token
            for token in tokens
            if token not in special_tokens
        ]

    return " ".join(tokens)
