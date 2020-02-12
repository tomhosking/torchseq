import re
# from bpemb import BPEmb
from utils.sentencepiece_pb2 import SentencePieceText

import unicodedata


# class BPE_bpemb:
#     _instance = None

#     pad_id = None
#     embedding_dim = None

#     @staticmethod
#     def instance():
#         if BPE._instance is None:
#             if BPE.pad_id is None:
#                 raise Exception('The vocab size hasnt been set for BPE!')
#             if BPE.embedding_dim is None:
#                 raise Exception('The vocab size hasnt been set for BPE!')
#             BPE._instance = BPEmb(lang="en", dim=BPE.embedding_dim, vs=BPE.pad_id, preprocess=True, add_pad_emb=True)
#         return BPE._instance

#     @staticmethod
#     def tokenise(text):
#         spt = SentencePieceText()
#         text = re.sub(r'[0-9]','0', text)
#         spt.ParseFromString(BPE.instance().spm.EncodeAsSerializedProto(text.lower()))

#         bos = [{'id': BPE.instance().BOS, 'text': BPE.instance().BOS_str, 'begin': 0, 'end': 0}]
#         eos = [{'id': BPE.instance().EOS, 'text': BPE.instance().EOS_str, 'begin': len(text), 'end': len(text)}]

#         return bos + [{'id': piece.id, 'text': piece.piece, 'begin': piece.begin, 'end': piece.end} for piece in spt.pieces] + eos


from transformers import  BertModel

from tokenizers import BertWordPieceTokenizer

class BPE:
    _instance = None

    pad_id = None
    embedding_dim = None
    bos_id = None
    eos_id = None

    model_slug = 'bert-base-uncased'

    

    @staticmethod
    def decode(token_id_tensor):
        return BPE.instance().decode(token_id_tensor.tolist(), skip_special_tokens=True)


    @staticmethod
    def instance():
        if BPE._instance is None:
            BPE._instance = tokenizer = BertWordPieceTokenizer("./data/bert-vocabs/{:}-vocab.txt".format(BPE.model_slug), lowercase=(BPE.model_slug[-8:] == '-uncased'))
            
            BPE.pad_id = BPE._instance.token_to_id('[PAD]')

            
            BPE.bos_id = BPE._instance.token_to_id('[CLS]')
            BPE.eos_id = BPE._instance.token_to_id('[SEP]')

            
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


        bos = [{'id': BPE.instance().token_to_id('[CLS]'), 'text': '[CLS]', 'begin': 0, 'end': 0}]
        eos = [{'id': BPE.instance().token_to_id('[SEP]'), 'text': '[SEP]', 'begin': len(text), 'end': len(text)}]

        tokenised = [{'id': token_ids[ix], 'text': token_texts[ix], 'begin': offsets[ix][0], 'end': offsets[ix][1]} for ix in range(len(output.tokens))]
        if add_bos_eos:
            return bos + tokenised + eos
        else:
            return tokenised

