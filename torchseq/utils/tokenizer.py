import torch

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from transformers import BartModel, BertModel, RobertaModel
from transformers import MBartTokenizer

from torchseq.utils.singleton import Singleton
from torchseq.utils.tokenizer_wordlevel import WordLevelTokenizer

FAIRSEQ_LANGUAGE_CODES = [
    "ar_AR",
    "cs_CZ",
    "de_DE",
    "en_XX",
    "es_XX",
    "et_EE",
    "fi_FI",
    "fr_XX",
    "gu_IN",
    "hi_IN",
    "it_IT",
    "ja_XX",
    "kk_KZ",
    "ko_KR",
    "lt_LT",
    "lv_LV",
    "my_MM",
    "ne_NP",
    "nl_XX",
    "ro_RO",
    "ru_RU",
    "si_LK",
    "tr_TR",
    "vi_VN",
    "zh_CN",
]


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

        if "mbart-" in model_slug:
            self.engine = MBartTokenizer.from_pretrained("sshleifer/mbart-large-cc25")

            self.pad_id = self.engine.pad_token_id
            self.mask_id = self.engine.mask_token_id
            self.unk_id = self.engine.unk_token_id

            self.bos_id = self.engine.bos_token_id
            self.eos_id = self.engine.eos_token_id

        elif "bart-" in model_slug or "roberta-" in model_slug:
            self.engine = ByteLevelBPETokenizer(
                "./data/pretrained-vocabs/{:}-vocab.json".format(model_slug.split("/")[-1]),
                "./data/pretrained-vocabs/{:}-merges.txt".format(model_slug.split("/")[-1]),
                lowercase=False,
            )

            self.engine.add_special_tokens(["<s>", "</s>", "<pad>", "<mask>", "<unk>"])

            self.pad_id = self.engine.token_to_id("<pad>")
            self.mask_id = self.engine.token_to_id("<mask>")
            self.unk_id = self.engine.token_to_id("<unk>")

            self.bos_id = self.engine.token_to_id("<s>")
            self.eos_id = self.engine.token_to_id("</s>")

        elif "ptb" in model_slug:
            self.engine = WordLevelTokenizer(
                "./data/pretrained-vocabs/{:}-vocab.json".format(model_slug.split("/")[-1])
            )

            self.pad_id = self.engine.token_to_id("<pad>")
            self.mask_id = self.engine.token_to_id("<mask>")
            self.unk_id = self.engine.token_to_id("<unk>")

            self.bos_id = self.engine.token_to_id("<s>")
            self.eos_id = self.engine.token_to_id("</s>")

        else:
            self.engine = BertWordPieceTokenizer(
                "./data/pretrained-vocabs/{:}-vocab.txt".format(model_slug.split("/")[-1]),
                lowercase=(model_slug[-8:] == "-uncased"),
            )

            self.pad_id = self.engine.token_to_id("[PAD]")
            self.mask_id = self.engine.token_to_id("[MASK]")
            self.unk_id = self.engine.token_to_id("[UNK")

            self.bos_id = self.engine.token_to_id("[CLS]")
            self.eos_id = self.engine.token_to_id("[SEP]")

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
        return torch.load("./data/pretrained-vocabs/{:}.embeddings.pt".format(model_slug.split("/")[-1]))

    def tokenise(self, text, add_bos_eos=True, src_lang_code=None, tgt_lang_code=None):
        output = Tokenizer().engine.encode(text)

        if "mbart-" in Tokenizer().model_slug:
            token_ids = output
            offsets = [(0, 0) for _ in range(len(token_ids))]
            token_texts = ["?" for _ in range(len(token_ids))]
        else:
            token_ids = output.ids
            offsets = output.offsets
            token_texts = output.tokens

        bos_str = "[CLS]" if "bert" in Tokenizer().model_slug else "<s>"
        eos_str = "[SEP]" if "bert" in Tokenizer().model_slug else "</s>"

        # mBART doesn't use bos tokens
        if "mbart-" in Tokenizer().model_slug:
            bos = (
                [{"id": self.engine.lang_code_to_id[tgt_lang_code], "text": tgt_lang_code, "begin": 0, "end": 0}]
                if tgt_lang_code is not None
                else []
            ) + [{"id": Tokenizer().bos_id, "text": bos_str, "begin": 0, "end": 0}]
            eos = (
                [
                    {
                        "id": self.engine.lang_code_to_id[src_lang_code],
                        "text": src_lang_code,
                        "begin": len(text),
                        "end": len(text),
                    }
                ]
                if src_lang_code is not None
                else []
            )
            # eos = []

        else:
            bos = [{"id": Tokenizer().bos_id, "text": bos_str, "begin": 0, "end": 0}]
            eos = [{"id": Tokenizer().eos_id, "text": eos_str, "begin": len(text), "end": len(text)}]

        if "bert-" in Tokenizer().model_slug:
            # NOTE: HF tokenizers automatically adds CLS/SEP tokens for BERT, so we have to fudge the indices to skip these
            tokenised = [
                {"id": token_ids[ix], "text": token_texts[ix], "begin": offsets[ix][0], "end": offsets[ix][1]}
                for ix in range(1, len(token_ids) - 1)
            ]
        else:
            tokenised = [
                {"id": token_ids[ix], "text": token_texts[ix], "begin": offsets[ix][0], "end": offsets[ix][1]}
                for ix in range(len(token_ids))
            ]

        if add_bos_eos:
            return bos + tokenised + eos
        else:
            return tokenised
