from utils.bpe_factory import BPE

class ParaphrasePair:
    def __init__(self, sent1_text, sent2_text):


        self._s1_doc = BPE.tokenise(sent1_text)
        self._s2_doc = BPE.tokenise(sent2_text)


    def s1_as_ids(self):
        id_list = [tok['id'] for tok in self._s1_doc]
        return id_list

    def s2_as_ids(self):
        id_list = [tok['id'] for tok in self._s2_doc]
        return id_list