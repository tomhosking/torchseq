from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from transformers import BartModel, BertModel, RobertaModel


class BPE:
    _instance = None

    pad_id = None
    embedding_dim = None
    bos_id = None
    eos_id = None
    mask_id = None
    unk_id = None

    model_slug = "bert-base-uncased"

    @staticmethod
    def reload():
        del BPE._instance
        BPE._instance = None
        return BPE.instance()

    @staticmethod
    def decode(token_id_tensor):
        return (
            BPE.instance().decode(token_id_tensor.tolist(), skip_special_tokens=True)
            # .replace(" ##", "")
            # .replace("# ", "#")
        )

    @staticmethod
    def instance():
        if BPE._instance is None:
            if "bart-" in BPE.model_slug or "roberta-" in BPE.model_slug:
                BPE._instance = ByteLevelBPETokenizer(
                    "./data/pretrained-vocabs/{:}-vocab.json".format(BPE.model_slug),
                    "./data/pretrained-vocabs/{:}-merges.txt".format(BPE.model_slug),
                    lowercase=False,
                )

                BPE._instance.add_special_tokens(["<s>", "</s>", "<pad>", "<mask>", "<unk>"])

                BPE.pad_id = BPE._instance.token_to_id("<pad>")
                BPE.mask_id = BPE._instance.token_to_id("<mask>")
                BPE.unk_id = BPE._instance.token_to_id("<unk>")

                BPE.bos_id = BPE._instance.token_to_id("<s>")
                BPE.eos_id = BPE._instance.token_to_id("</s>")

                if "bart-" in BPE.model_slug:
                    model = BartModel.from_pretrained(BPE.model_slug)
                    BPE._instance.embeddings = model.encoder.embed_tokens.weight.data

                    del model
                elif "roberta-" in BPE.model_slug:
                    model = RobertaModel.from_pretrained(BPE.model_slug)
                    BPE._instance.embeddings = model.embeddings.word_embeddings.weight.data

                    del model
            else:
                BPE._instance = BertWordPieceTokenizer(
                    "./data/pretrained-vocabs/{:}-vocab.txt".format(BPE.model_slug),
                    lowercase=(BPE.model_slug[-8:] == "-uncased"),
                )

                BPE.pad_id = BPE._instance.token_to_id("[PAD]")
                BPE.mask_id = BPE._instance.token_to_id("[MASK]")
                BPE.unk_id = BPE._instance.token_to_id("[UNK")

                BPE.bos_id = BPE._instance.token_to_id("[CLS]")
                BPE.eos_id = BPE._instance.token_to_id("[SEP]")

                model = BertModel.from_pretrained(BPE.model_slug)
                BPE._instance.embeddings = model.embeddings.word_embeddings.weight.data

                del model

        return BPE._instance

    @staticmethod
    def tokenise(text, add_bos_eos=True):
        output = BPE.instance().encode(text)

        token_ids = output.ids
        offsets = output.offsets
        token_texts = output.tokens

        bos_str = "[CLS]" if "bert" in BPE.model_slug else "<s>"
        eos_str = "[SEP]" if "bert" in BPE.model_slug else "</s>"

        bos = [{"id": BPE.bos_id, "text": bos_str, "begin": 0, "end": 0}]
        eos = [{"id": BPE.eos_id, "text": eos_str, "begin": len(text), "end": len(text)}]

        if "bert-" in BPE.model_slug:
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
