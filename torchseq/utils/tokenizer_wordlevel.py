from tokenizers.implementations.base_tokenizer import BaseTokenizer
from typing import Optional, List, Union

from tokenizers import Tokenizer, Encoding, AddedToken
from tokenizers.models import WordLevel
from tokenizers.normalizers import unicode_normalizer_from_str, Lowercase, Sequence
import tokenizers


class WordLevelTokenizer(BaseTokenizer):
    """WordLevelTokenizer
    Represents a simple word level tokenization
    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: Union[str, AddedToken] = "<unk>",
        bos_token: Union[str, AddedToken] = "<s>",
        eos_token: Union[str, AddedToken] = "</s>",
        pad_token: Union[str, AddedToken] = "<pad>",
        mask_token: Union[str, AddedToken] = "<mask>",
        lowercase: bool = False,
        unicode_normalizer: Optional[str] = None,
    ):
        if vocab_file is not None:
            print(vocab_file)
            tokenizer = Tokenizer(WordLevel(vocab_file))
        else:
            tokenizer = Tokenizer(WordLevel())

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(bos_token)) is not None:
            tokenizer.add_special_tokens([str(bos_token)])
        if tokenizer.token_to_id(str(eos_token)) is not None:
            tokenizer.add_special_tokens([str(eos_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()

        if vocab_file is not None:
            bos_token_id = tokenizer.token_to_id(str(bos_token))
            if bos_token_id is None:
                raise TypeError("bos_token not found in the vocabulary")
            eos_token_id = tokenizer.token_to_id(str(eos_token))
            if eos_token_id is None:
                raise TypeError("eos_token not found in the vocabulary")

            # tokenizer.post_processor = tokenizers.processors.BertProcessing(
            #     (str(bos_token), bos_token_id), (str(eos_token), eos_token_id)
            # )

        parameters = {
            "model": "WordLevel",
            "unk_token": unk_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "lowercase": lowercase,
            "unicode_normalizer": unicode_normalizer,
        }

        super().__init__(tokenizer, parameters)
