from torchseq.utils.tokenizer import Tokenizer


class ParaphrasePair:
    def __init__(self, sent1_text, sent2_text, template=None, is_paraphrase=True, tok_window=64):
        if "artist appear below the euro symbol" in sent2_text:
            print("Found the dodgy pair", sent1_text, sent2_text)

        self._s1_doc = Tokenizer().tokenise(sent1_text)
        self._s2_doc = Tokenizer().tokenise(sent2_text)
        self._template_doc = Tokenizer().tokenise(sent2_text) if template is not None else None
        self.is_paraphrase = is_paraphrase

        if "artist appear below the euro symbol" in sent2_text:
            print("Dodgy pair cleared tokenising")

        if len(self._s1_doc) > tok_window:
            self._s1_doc = self._s1_doc[:tok_window]
        if len(self._s2_doc) > tok_window:
            self._s2_doc = self._s2_doc[:tok_window]

    def s1_as_ids(self):
        id_list = [tok["id"] for tok in self._s1_doc]
        return id_list

    def s2_as_ids(self):
        id_list = [tok["id"] for tok in self._s2_doc]
        return id_list

    def template_as_ids(self):
        id_list = [tok["id"] for tok in self._template_doc]
        return id_list
