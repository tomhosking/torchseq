from nltk.tokenize import sent_tokenize

from args import FLAGS

from datasets.loaders import OOV, SOS, EOS, PAD

from utils.bpe_factory import BPE


# Convert a text context-question-answer triple to a cropped tokenised encoding
class CQATriple:
    def __init__(self, context, answer, a_pos, question = None):

        # How many sentences either side of the answer should we keep?
        # TODO: move to config (how to pass config into this constructor?)
        SENT_WINDOW_SIZE = 0

        # Max num tokens either size of the answer
        TOKEN_WINDOW_SIZE = 200
        
        self.context_text = context
        self.is_training = question is not None
        self.question_text = question
        self.answer_text = answer
        self.a_char_pos_uncrop = a_pos

        ctxt_sents = sent_tokenize(self.context_text)
        ctxt_char_offsets = []
        for i,sent in enumerate(ctxt_sents):
            sent_char_offset = self.context_text.find(sent, ctxt_char_offsets[i-1][1]) if i>0 else 0
            ctxt_char_offsets.append( (sent_char_offset, sent_char_offset+len(sent)) )

        # self._ctxt_doc_uncrop = BPE.tokenise(self.context_text)
        ctxt_sent_toks = [BPE.tokenise(sent) for sent in ctxt_sents]
        
        self._ans_doc = BPE.tokenise(self.answer_text)
        self._q_doc = BPE.tokenise(self.question_text) if self.is_training else None

        # Find the answer in the uncropped context
        self.a_tok_pos_uncrop = None
        for sent_ix, sent_toks in enumerate(ctxt_sent_toks):
            offset = sum([len(sent)+1 for sent in ctxt_sents[:sent_ix]])
            # print(sent_ix, offset)
            
            for ix,tok in enumerate(sent_toks):
                # print(tok['text'], tok['begin']+offset, tok['end']+offset, self.a_char_pos_uncrop)
                if self.a_char_pos_uncrop >= tok['begin']+offset and self.a_char_pos_uncrop <= tok['end']+offset:
                    self.a_tok_pos_uncrop = ix + sum([len(sent) for sent in ctxt_sent_toks[:sent_ix]])
                    # print(tok, ix, sent_ix, [len(sent) for sent in ctxt_sent_toks[:sent_ix]])
                    break

        if self.a_tok_pos_uncrop is None:
            raise Exception('Couldnt find the answer token position')

        # Find the sentence that contains the answer
        for ix,offset in enumerate(ctxt_char_offsets):
            if self.a_char_pos_uncrop >= offset[0] and self.a_char_pos_uncrop < offset[1]:
                self.a_sent_idx = ix
                break
                
        
        
        self.cropped_sents = ctxt_sent_toks[max(0,self.a_sent_idx-SENT_WINDOW_SIZE):min(len(ctxt_sents), self.a_sent_idx+SENT_WINDOW_SIZE+1)]
        self._ctxt_doc = [tok for sent in self.cropped_sents for tok in sent]
        self._ctxt_doc_uncrop = [tok for sent in ctxt_sent_toks for tok in sent]
        # self._ctxt_doc = self._ctxt_doc_uncrop # TODO: reimplement sentence cropping

        # print(self.answer_text, self._ctxt_doc_uncrop[self.a_tok_pos_uncrop:self.a_tok_pos_uncrop+3])
        # print(self.context_text[self.a_char_pos_uncrop:self.a_char_pos_uncrop+10])

        # self.a_pos = self.a_char_pos_uncrop# - self.cropped_sents[0].start_char
        self.a_tok_pos = self.a_tok_pos_uncrop - sum([len(sent) for sent in ctxt_sent_toks[:max(0,self.a_sent_idx-SENT_WINDOW_SIZE)]])
        self.a_tok_end = self.a_tok_pos + len(self._ans_doc)-2 # subtract two because of the BOS/SOS tags

        # print(self.a_tok_pos_uncrop, self.a_sent_idx, self.a_tok_pos, self.a_tok_end)
        # print(self._ctxt_doc[self.a_tok_pos:self.a_tok_end])
        # print(self._ans_doc)
        # exit()

        # TODO: crop it...
        if len(self._ctxt_doc) > TOKEN_WINDOW_SIZE:
            # print('OVERSIZE CONTEXT!', len(self._ctxt_doc))
            crop_begin_ix = max(self.a_tok_pos-(TOKEN_WINDOW_SIZE//2), 0)
            crop_end_ix = min(self.a_tok_pos+(TOKEN_WINDOW_SIZE//2), len(self._ctxt_doc))
            # print(crop_begin_ix, crop_end_ix, self.a_tok_pos)
            self.a_tok_pos = self.a_tok_pos - crop_begin_ix
            self.a_tok_end = self.a_tok_end - crop_begin_ix
            self._ctxt_doc = self._ctxt_doc[crop_begin_ix:crop_end_ix]
            
            # print(len(self._ctxt_doc))


        # print(self.context_text)
        # print(BPE.instance().decode_ids([tok['id'] for tok in self._ctxt_doc]))
        # print(self.answer_text)
        # exit()

        # Determine the context specific vocab for this example, using only novel words
        # self.copy_vocab = {w:i+len(self.vocab) for i,w in enumerate(list(set([tok['text'] for tok in self._ctxt_doc if tok['text'] not in self.vocab.keys()]))) }


    # def lookup_vocab(self, tok):
    #     if tok in self.vocab.keys():
    #         return self.vocab[tok]
    #     elif tok in self.copy_vocab.keys():
    #         return self.copy_vocab[tok]
    #     else:
    #         return self.vocab[OOV]

    def ctxt_as_ids(self):
        id_list = [tok['id'] for tok in self._ctxt_doc]
        return id_list

    def q_as_ids(self):
        id_list = [tok['id'] for tok in self._q_doc] if self.is_training else None
        return id_list

    def ans_as_ids(self):
        id_list = [tok['id'] for tok in self._ans_doc]
        return id_list

    def ctxt_as_bio(self):
        id_list = [1 if ix == self.a_tok_pos else 2 if ix > self.a_tok_pos and ix <= self.a_tok_end else 0 for ix in range(len(self._ctxt_doc))]
        return id_list
