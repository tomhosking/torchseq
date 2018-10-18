import spacy

from args import FLAGS

from datasets.loaders import OOV, SOS, EOS, PAD


# Convert a text context-question-answer triple to a cropped tokenised encoding
class CQATriple:
    def __init__(self, vocab, context, answer, a_pos, question = None):
        self.vocab = vocab
        self.context_text = context
        self.is_training = question is not None
        self.question_text = question
        self.answer_text = answer
        self.a_pos_uncrop = a_pos

        nlp = spacy.load('en')
        self._ctxt_doc_uncrop = nlp(self.context_text)
        self._ans_doc = nlp(self.answer_text)
        self._q_doc = nlp(self.question_text) if self.is_training else None

        # Find the answer in the uncropped context
        for tok in self._ctxt_doc_uncrop:
            if self.a_pos_uncrop >= tok.idx and self.a_pos_uncrop < tok.idx + len(tok):
                self.a_tok_pos_uncrop = tok.i

        # Now crop the context, and record the adjusted positions
        all_sents = []
        for ix,sent in enumerate(self._ctxt_doc_uncrop.sents):
            if self.a_tok_pos_uncrop >= sent.start and self.a_tok_pos_uncrop < sent.end + len(sent):
                self.a_sent_idx = ix
            all_sents.append(sent)

        sentence_window_before = FLAGS.crop_sentences_before
        sentence_window_after = FLAGS.crop_sentences_after

        self.cropped_sents = all_sents[max(0,self.a_sent_idx-sentence_window_before):min(len(all_sents), self.a_sent_idx+sentence_window_after+1)]
        self._ctxt_doc = [tok for sent in self.cropped_sents for tok in sent]

        self.a_pos = self.a_pos_uncrop - self.cropped_sents[0].start_char
        self.a_tok_pos = self.a_tok_pos_uncrop - self.cropped_sents[0].start
        self.a_tok_end = self.a_tok_pos + len(self._ans_doc)-1

        # Determine the context specific vocab for this example, using only novel words
        self.copy_vocab = {w:i+len(self.vocab) for i,w in enumerate(list(set([tok.text for tok in self._ctxt_doc if tok.text not in self.vocab.keys()]))) }


    def lookup_vocab(self, tok):
        if tok in self.vocab.keys():
            return self.vocab[tok]
        elif tok in self.copy_vocab.keys():
            return self.copy_vocab[tok]
        else:
            return self.vocab[OOV]

    def ctxt_as_ids(self):
        id_list = [self.lookup_vocab(tok.text) for tok in self._ctxt_doc]
        return id_list

    def q_as_ids(self):
        id_list = [self.lookup_vocab(tok.text) for tok in self._q_doc] if self.is_training else None
        return id_list

    def ans_as_ids(self):
        id_list = [self.lookup_vocab(tok.text) for tok in self._ans_doc]
        return id_list

    def ctxt_as_bio(self):
        id_list = [1 if ix == self.a_tok_pos else 2 if ix > self.a_tok_pos and ix <= self.a_tok_end else 0 for ix in range(len(self._ctxt_doc))]
        return id_list
