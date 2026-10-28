"""Token decoding."""
from typing import List, Optional
from .vocab import Vocabulary
from config.constants import PAD_TOKEN_ID, UNK_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID


def decode(ids: List[int], vocab: Optional[Vocabulary] = None, skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs back to a string.
    """
    if vocab is None:
        vocab = Vocabulary()

    tokens = [vocab.id_to_token(int(i)) for i in ids]

    if skip_special_tokens:
        special_ids = {
            getattr(vocab, "pad_token_id", PAD_TOKEN_ID),
            getattr(vocab, "unk_token_id", UNK_TOKEN_ID),
            getattr(vocab, "bos_token_id", BOS_TOKEN_ID),
            getattr(vocab, "eos_token_id", EOS_TOKEN_ID),
        }
        tokens = [token for token_id, token in zip(ids, tokens) if int(token_id) not in special_ids]

    return " ".join(tokens)
