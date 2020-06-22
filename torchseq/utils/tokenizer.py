import torch

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from transformers import BartModel, BertModel, RobertaModel


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Tokenizer(metaclass=Singleton):

    # class __Tokenizer:
    pad_id = None
    embedding_dim = None
    bos_id = None
    eos_id = None
    mask_id = None
    unk_id = None

    model_slug = None

    engine = None

    def __init__(self, model_slug=None):
        if model_slug is None:
            raise Exception("Tokenizer needs to be initialized with a model name before use!")

        self.model_slug = model_slug

        if "bart-" in model_slug or "roberta-" in model_slug:
            self.engine = ByteLevelBPETokenizer(
                "./data/pretrained-vocabs/{:}-vocab.json".format(model_slug),
                "./data/pretrained-vocabs/{:}-merges.txt".format(model_slug),
                lowercase=False,
            )

            self.engine.add_special_tokens(["<s>", "</s>", "<pad>", "<mask>", "<unk>"])

            self.pad_id = self.engine.token_to_id("<pad>")
            self.mask_id = self.engine.token_to_id("<mask>")
            self.unk_id = self.engine.token_to_id("<unk>")

            self.bos_id = self.engine.token_to_id("<s>")
            self.eos_id = self.engine.token_to_id("</s>")

        else:
            self.engine = BertWordPieceTokenizer(
                "./data/pretrained-vocabs/{:}-vocab.txt".format(model_slug), lowercase=(model_slug[-8:] == "-uncased"),
            )

            self.pad_id = self.engine.token_to_id("[PAD]")
            self.mask_id = self.engine.token_to_id("[MASK]")
            self.unk_id = self.engine.token_to_id("[UNK")

            self.bos_id = self.engine.token_to_id("[CLS]")
            self.eos_id = self.engine.token_to_id("[SEP]")

            print(self.model_slug)
            print(self.bos_id)

    # instance = None
    # def __init__(self, model_slug=None):
    #     if not Tokenizer.instance and model_slug is not None:
    #         Tokenizer.instance = Tokenizer.__Tokenizer(model_slug)
    #     elif model_slug is None:
    #         raise Exception("Tokenizer needs to be initialized with a model name before use!")

    # def __getattr__(self, name):
    #     print(name)
    #     return getattr(Tokenizer.instance, name)

    def reload(self, model_slug):
        del Tokenizer._instances[Tokenizer]
        Tokenizer(model_slug)

    def decode(self, token_id_tensor):
        return (
            Tokenizer().engine.decode(token_id_tensor.tolist(), skip_special_tokens=True)
            # .replace(" ##", "")
            # .replace("# ", "#")
        )

    def get_embeddings(self, model_slug):
        return torch.load("./data/pretrained-vocabs/{:}.embeddings.pt".format(model_slug))

    def tokenise(self, text, add_bos_eos=True):
        output = Tokenizer().engine.encode(text)

        token_ids = output.ids
        offsets = output.offsets
        token_texts = output.tokens

        bos_str = "[CLS]" if "bert" in Tokenizer().model_slug else "<s>"
        eos_str = "[SEP]" if "bert" in Tokenizer().model_slug else "</s>"

        bos = [{"id": Tokenizer().bos_id, "text": bos_str, "begin": 0, "end": 0}]
        eos = [{"id": Tokenizer().eos_id, "text": eos_str, "begin": len(text), "end": len(text)}]

        if "bert-" in Tokenizer().model_slug:
            # NOTE: HF tokenizers automatically adds CLS/SEP tokens for BERT, so we have to fudge the indices to skip these
            tokenised = [
                {"id": token_ids[ix], "text": token_texts[ix], "begin": offsets[ix][0], "end": offsets[ix][1]}
                for ix in range(1, len(output.tokens) - 1)
            ]
        else:
            tokenised = [
                {"id": token_ids[ix], "text": token_texts[ix], "begin": offsets[ix][0], "end": offsets[ix][1]}
                for ix in range(len(output.tokens))
            ]

        if add_bos_eos:
            return bos + tokenised + eos
        else:
            return tokenised
