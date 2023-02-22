import os
import torch

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers import Tokenizer as HFTokenizer
from transformers import BartModel, BertModel, RobertaModel
from transformers import MBart50TokenizerFast

from torchseq.utils.singleton import Singleton
from torchseq.utils.tokenizer_wordlevel import WordLevelTokenizer

# self.data_path = "./data/"

# FAIRSEQ_LANGUAGE_CODES = [
#     "ar_AR",
#     "cs_CZ",
#     "de_DE",
#     "en_XX",
#     "es_XX",
#     "et_EE",
#     "fi_FI",
#     "fr_XX",
#     "gu_IN",
#     "hi_IN",
#     "it_IT",
#     "ja_XX",
#     "kk_KZ",
#     "ko_KR",
#     "lt_LT",
#     "lv_LV",
#     "my_MM",
#     "ne_NP",
#     "nl_XX",
#     "ro_RO",
#     "ru_RU",
#     "si_LK",
#     "tr_TR",
#     "vi_VN",
#     "zh_CN",
# ]

FAIRSEQ_LANGUAGE_CODES = {  # NOTE(SS): resize embeddings will break this
    "ar_AR": 250001,
    "cs_CZ": 250002,
    "de_DE": 250003,
    "en_XX": 250004,
    "es_XX": 250005,
    "et_EE": 250006,
    "fi_FI": 250007,
    "fr_XX": 250008,
    "gu_IN": 250009,
    "hi_IN": 250010,
    "it_IT": 250011,
    "ja_XX": 250012,
    "kk_KZ": 250013,
    "ko_KR": 250014,
    "lt_LT": 250015,
    "lv_LV": 250016,
    "my_MM": 250017,
    "ne_NP": 250018,
    "nl_XX": 250019,
    "ro_RO": 250020,
    "ru_RU": 250021,
    "si_LK": 250022,
    "tr_TR": 250023,
    "vi_VN": 250024,
    "zh_CN": 250025,
    "af_ZA": 250026,
    "az_AZ": 250027,
    "bn_IN": 250028,
    "fa_IR": 250029,
    "he_IL": 250030,
    "hr_HR": 250031,
    "id_ID": 250032,
    "ka_GE": 250033,
    "km_KH": 250034,
    "mk_MK": 250035,
    "ml_IN": 250036,
    "mn_MN": 250037,
    "mr_IN": 250038,
    "pl_PL": 250039,
    "ps_AF": 250040,
    "pt_XX": 250041,
    "sv_SE": 250042,
    "sw_KE": 250043,
    "ta_IN": 250044,
    "te_IN": 250045,
    "th_TH": 250046,
    "tl_XX": 250047,
    "uk_UA": 250048,
    "ur_PK": 250049,
    "xh_ZA": 250050,
    "gl_ES": 250051,
    "sl_SI": 250052,
}


class Tokenizer:
    pad_id: int
    embedding_dim: int
    bos_id: int
    eos_id: int
    mask_id: int
    unk_id: int

    model_slug = None

    engine = None

    def __init__(self, model_slug, data_path="./data/"):
        if model_slug is None:
            raise Exception("Tokenizer needs to be initialized with a model name before use!")

        self.model_slug = model_slug
        self.data_path = data_path

        # Most tokenizers should come with pretrained embeddings
        self.has_embeddings = True

        if "mbart-" in model_slug:
            self.engine = MBart50TokenizerFast.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="en_XX", add_prefix_space=True
            )

            self.pad_id = self.engine.pad_token_id
            self.mask_id = self.engine.mask_token_id
            self.unk_id = self.engine.unk_token_id

            self.bos_id = self.engine.bos_token_id
            self.eos_id = self.engine.eos_token_id

        elif "bart-" in model_slug or "roberta-" in model_slug:
            self.engine = ByteLevelBPETokenizer.from_file(
                os.path.join(self.data_path, "pretrained-vocabs/{:}-vocab.json".format(model_slug.split("/")[-1])),
                os.path.join(self.data_path, "pretrained-vocabs/{:}-merges.txt".format(model_slug.split("/")[-1])),
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
                os.path.join(self.data_path, "pretrained-vocabs/{:}-vocab.json".format(model_slug.split("/")[-1]))
            )

            self.pad_id = self.engine.token_to_id("<pad>")
            self.mask_id = self.engine.token_to_id("<mask>")
            self.unk_id = self.engine.token_to_id("<unk>")

            self.bos_id = self.engine.token_to_id("<s>")
            self.eos_id = self.engine.token_to_id("</s>")

            self.has_embeddings = False

        elif "wordlevel" in model_slug:
            vocab_file = model_slug.replace("wordlevel:", "")
            self.engine = WordLevelTokenizer(os.path.join(self.data_path, vocab_file))

            self.pad_id = self.engine.token_to_id("<pad>")
            self.mask_id = self.engine.token_to_id("<mask>")
            self.unk_id = self.engine.token_to_id("<unk>")

            self.bos_id = self.engine.token_to_id("<s>")
            self.eos_id = self.engine.token_to_id("</s>")

            self.has_embeddings = False

        elif model_slug[:5] == "bert-":
            self.engine = BertWordPieceTokenizer.from_file(
                os.path.join(self.data_path, "pretrained-vocabs/{:}-vocab.txt".format(model_slug.split("/")[-1])),
                lowercase=(model_slug[-8:] == "-uncased"),
            )

            self.pad_id = self.engine.token_to_id("[PAD]")
            self.mask_id = self.engine.token_to_id("[MASK]")
            self.unk_id = self.engine.token_to_id("[UNK")

            self.bos_id = self.engine.token_to_id("[CLS]")
            self.eos_id = self.engine.token_to_id("[SEP]")
        else:
            self.engine = HFTokenizer.from_pretrained(model_slug)

            # TODO: How can we generically work out what the special tokens are? Otherwise they need to be passed in via config

            self.pad_id = self.engine.token_to_id("<pad>")
            self.mask_id = self.engine.token_to_id("<mask>")
            self.unk_id = self.engine.token_to_id("<unk>")

            self.bos_id = self.engine.token_to_id("<s>")
            self.eos_id = self.engine.token_to_id("</s>")

        # Vocab size from PretrainedFastTokenize is __len__ attr
        if "mbart-" in model_slug:
            self.vocab_size = len(self.engine)
        else:
            self.vocab_size = self.engine.get_vocab_size()

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
        return self.engine.decode(token_id_tensor.tolist(), skip_special_tokens=True)

    def get_embeddings(self):
        return torch.load(
            os.path.join(self.data_path, "pretrained-vocabs/{:}.embeddings.pt".format(self.model_slug.split("/")[-1]))
        ).cpu()

    def tokenise(self, text, add_bos_eos=True, src_lang=None, tgt_lang=None):

        if "mbart-" in self.model_slug:
            output = self.engine(text, return_offsets_mapping=True, add_special_tokens=False)

            token_ids = output["input_ids"]
            offsets = output["offset_mapping"]
            token_texts = output.tokens()  # tokens() is a method in PretrainedFastTokenizer
        else:
            output = self.engine.encode(text)

            token_ids = output.ids
            offsets = output.offsets
            token_texts = output.tokens

        bos_str = "[CLS]" if "bert" in self.model_slug else "<s>"
        eos_str = "[SEP]" if "bert" in self.model_slug else "</s>"

        # mBART doesn't use bos tokens
        if "mbart-" in self.model_slug:
            eos = [{"id": self.eos_id, "text": eos_str, "begin": len(text), "end": len(text)}]
            bos = []

        else:
            bos = [{"id": self.bos_id, "text": bos_str, "begin": 0, "end": 0}]
            eos = [{"id": self.eos_id, "text": eos_str, "begin": len(text), "end": len(text)}]

        if "bert-" in self.model_slug:
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
