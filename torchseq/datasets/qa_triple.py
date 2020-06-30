import unicodedata

from nltk.tokenize import sent_tokenize


from torchseq.utils.tokenizer import Tokenizer


# This fn converts character positions to byte positions - why? Because SentencePiece works in terms of byte offsets, and SQuAD contains some full width characters!
def get_byte_offsets(text, character_offset):
    return character_offset
    # needs_conversion = len(text) != len(text.encode('utf8'))
    # if needs_conversion:
    #     begin_offset = len(text[:character_offset].encode('utf8'))
    # else:
    #     begin_offset = character_offset
    # return begin_offset


# Convert a text context-question-answer triple to a cropped tokenised encoding
class QATriple:
    def __init__(self, context, answer, a_pos, question=None, sent_window=0, tok_window=300, o_tag=1):

        self.o_tag = o_tag

        # How many sentences either side of the answer should we keep?
        # TODO: move to config (how to pass config into this constructor?)
        SENT_WINDOW_SIZE = sent_window

        # Max num tokens either size of the answer
        TOKEN_WINDOW_SIZE = tok_window

        def _run_strip_accents(text):
            # nfkd_form = unicodedata.normalize('NFKD', text)
            # return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
            text = unicodedata.normalize("NFD", text)
            output = []
            for char in text:
                cat = unicodedata.category(char)
                if cat == "Mn":
                    continue
                else:
                    output.append(char)
            return "".join(output)

        def _is_whitespace(char):
            """Checks whether `chars` is a whitespace character."""
            # \t, \n, and \r are technically contorl characters but we treat them
            # as whitespace since they are generally considered as such.
            if char == " " or char == "\t" or char == "\n" or char == "\r":
                return True
            cat = unicodedata.category(char)
            if cat == "Zs":
                return True
            return False

        def _is_control(char):
            """Checks whether `chars` is a control character."""
            # These are technically control characters but we count them as whitespace
            # characters.
            if char == "\t" or char == "\n" or char == "\r":
                return False
            cat = unicodedata.category(char)
            if cat.startswith("C"):
                return True
            return False

        def _clean_text(text):
            """Performs invalid character removal and whitespace cleanup on text."""
            output = []
            for char in text:
                cp = ord(char)
                if cp == 0 or cp == 0xFFFD or _is_control(char):
                    continue
                if _is_whitespace(char):
                    output.append(" ")
                else:
                    output.append(char)
            return "".join(output)

        def normalise(text):
            return text
            clean_text = _run_strip_accents(text)
            clean_text = _clean_text(clean_text)
            return clean_text
            # return text.replace(chr(8211), '-')

        self.context_text = normalise(context)
        self.is_training = question is not None
        self.question_text = normalise(question)
        self.answer_text = normalise(answer)
        self.a_char_pos_uncrop = get_byte_offsets(self.context_text, a_pos)
        self.a_char_end_uncrop = self.a_char_pos_uncrop + len(self.answer_text)  # .encode('utf8'))

        # print(self.a_char_pos_uncrop, a_pos, "->", self.a_char_end_uncrop, a_pos +len(answer))

        # if self.a_char_pos_uncrop - a_pos < -10:
        #     print(a_pos)
        #     print(context)

        ctxt_sents = sent_tokenize(self.context_text)
        ctxt_char_offsets = []
        offset = 0
        for i, sent in enumerate(ctxt_sents):
            sent_char_offset = self.context_text.find(sent, offset) if i > 0 else 0  # ctxt_char_offsets[i-1][1]-1

            offset = sent_char_offset + len(sent)
            sent_char_offset = get_byte_offsets(self.context_text, sent_char_offset)

            ctxt_char_offsets.append((sent_char_offset, sent_char_offset + len(sent)))  # .encode('utf8')

        # self._ctxt_doc_uncrop = Tokenizer().tokenise(self.context_text)
        ctxt_sent_toks = [Tokenizer().tokenise(sent) for sent in ctxt_sents]

        self._ans_doc = Tokenizer().tokenise(self.answer_text)
        self._q_doc = Tokenizer().tokenise(self.question_text, add_bos_eos=True) if self.is_training else None

        # Find the answer in the uncropped context
        self.a_tok_pos_uncrop = None
        self.a_tok_end_uncrop = None
        for sent_ix, sent_toks in enumerate(ctxt_sent_toks):
            # offset = sum([len(sent)+1 for sent in ctxt_sents[:sent_ix]])
            offset = ctxt_char_offsets[sent_ix][0]
            # print(sent_ix, offset)

            for ix, tok in enumerate(sent_toks):
                if self.a_char_pos_uncrop >= tok["begin"] + offset:  # and self.a_char_pos_uncrop <= tok['end']+offset
                    self.a_tok_pos_uncrop = ix + sum([len(sent) for sent in ctxt_sent_toks[:sent_ix]])

                if self.a_char_end_uncrop >= tok["begin"] + offset and self.a_char_end_uncrop <= tok["end"] + offset:
                    self.a_tok_end_uncrop = ix + sum([len(sent) for sent in ctxt_sent_toks[:sent_ix]])
                    break

                if self.a_char_end_uncrop <= tok["begin"] + offset and self.a_tok_end_uncrop is None:
                    self.a_tok_end_uncrop = ix + sum([len(sent) for sent in ctxt_sent_toks[:sent_ix]]) - 1
                    break

        if self.a_tok_pos_uncrop is None:
            print("Cannot find a_tok_pos")
            print(context)
            print(a_pos, self.a_char_pos_uncrop)
            print(self.answer_text)
            print(ctxt_char_offsets)
            raise Exception("Couldnt find the answer token position")
        if self.a_tok_end_uncrop is None:
            print("Cannot find a_tok_end")
            print(context)
            print(a_pos, self.a_char_end_uncrop)
            print(self.answer_text)
            print(ctxt_char_offsets)
            raise Exception("Couldnt find the answer token END position")

        # Find the sentence that contains the answer
        self.a_sent_idx = None
        for ix, offset in enumerate(ctxt_char_offsets):
            if self.a_char_pos_uncrop >= offset[0] and self.a_char_pos_uncrop < offset[1]:
                self.a_sent_idx = ix
                break

        if self.a_sent_idx is None:
            print("Couldnt find sentence idx")
            print(ctxt_char_offsets)
            print(self.a_char_pos_uncrop)
            print(self.answer_text)
            print(self.context_text)
            exit()

        self.cropped_sents = (
            ctxt_sent_toks[
                max(0, self.a_sent_idx - SENT_WINDOW_SIZE) : min(
                    len(ctxt_sents), self.a_sent_idx + SENT_WINDOW_SIZE + 1
                )
            ]
            if SENT_WINDOW_SIZE is not None
            else ctxt_sent_toks
        )
        self._ctxt_doc = [tok for sent in self.cropped_sents for tok in sent]
        self._ctxt_doc_uncrop = [tok for sent in ctxt_sent_toks for tok in sent]

        self.a_tok_pos = self.a_tok_pos_uncrop - (
            sum([len(sent) for sent in ctxt_sent_toks[: max(0, self.a_sent_idx - SENT_WINDOW_SIZE)]])
            if SENT_WINDOW_SIZE is not None
            else 0
        )
        # self.a_tok_end = self.a_tok_pos + len(self._ans_doc)-3 - (1 if self._ans_doc[1]['text'] == '▁' else 0) # subtract two because of the BOS/SOS tags, one because the ix should be inclusive, and potentially one because of the tokenisation mismatch between a standalone answer and an answer in the middle of a sentence
        self.a_tok_end = self.a_tok_end_uncrop - (
            sum([len(sent) for sent in ctxt_sent_toks[: max(0, self.a_sent_idx - SENT_WINDOW_SIZE)]])
            if SENT_WINDOW_SIZE is not None
            else 0
        )

        if TOKEN_WINDOW_SIZE is not None and len(self._ctxt_doc) > TOKEN_WINDOW_SIZE:
            # print('OVERSIZE CONTEXT!', len(self._ctxt_doc))
            crop_begin_ix = max(self.a_tok_pos - (TOKEN_WINDOW_SIZE // 2), 0)
            crop_end_ix = min(self.a_tok_pos + (TOKEN_WINDOW_SIZE // 2), len(self._ctxt_doc))
            # print(crop_begin_ix, crop_end_ix, self.a_tok_pos)
            self.a_tok_pos = self.a_tok_pos - crop_begin_ix
            self.a_tok_end = self.a_tok_end - crop_begin_ix
            self._ctxt_doc = self._ctxt_doc[crop_begin_ix:crop_end_ix]

    def ctxt_as_ids(self):
        id_list = [tok["id"] for tok in self._ctxt_doc]
        return id_list

    def q_as_ids(self):
        id_list = [tok["id"] for tok in self._q_doc] if self.is_training else None
        return id_list

    def ans_as_ids(self):
        id_list = [tok["id"] for tok in self._ans_doc]
        return id_list

    def ctxt_as_bio(self):
        id_list = [
            1 if ix == self.a_tok_pos else self.o_tag if ix > self.a_tok_pos and ix <= self.a_tok_end else 0
            for ix in range(len(self._ctxt_doc))
        ]
        return id_list
