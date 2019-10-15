import re
from bpemb import BPEmb
from utils.sentencepiece_pb2 import SentencePieceText

class BPE:
    _instance = None

    pad_id = None
    embedding_dim = None

    @staticmethod
    def instance():
        if BPE._instance is None:
            if BPE.pad_id is None:
                raise Exception('The vocab size hasnt been set for BPE!')
            if BPE.embedding_dim is None:
                raise Exception('The vocab size hasnt been set for BPE!')
            BPE._instance = BPEmb(lang="en", dim=BPE.embedding_dim, vs=BPE.pad_id, preprocess=True, add_pad_emb=True)
        return BPE._instance

    @staticmethod
    def tokenise(text):
        spt = SentencePieceText()
        text = re.sub(r'[0-9]','0', text)
        spt.ParseFromString(BPE.instance().spm.EncodeAsSerializedProto(text.lower()))

        bos = [{'id': BPE.instance().BOS, 'text': BPE.instance().BOS_str, 'begin': 0, 'end': 0}]
        eos = [{'id': BPE.instance().EOS, 'text': BPE.instance().EOS_str, 'begin': len(text), 'end': len(text)}]

        return bos + [{'id': piece.id, 'text': piece.piece, 'begin': piece.begin, 'end': piece.end} for piece in spt.pieces] + eos

