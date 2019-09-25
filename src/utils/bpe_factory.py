
from bpemb import BPEmb
from utils.sentencepiece_pb2 import SentencePieceText

class BPE:
    _instance = None

    @staticmethod
    def instance():
        if BPE._instance is None:
            # TODO: dim is hardcoded!
            BPE._instance = BPEmb(lang="en", dim=200, vs=10000, preprocess=False, add_pad_emb=True)
        return BPE._instance

    @staticmethod
    def tokenise(text):
        spt = SentencePieceText()
        spt.ParseFromString(BPE.instance().spm.EncodeAsSerializedProto(text.lower()))

        return [{'id': piece.id, 'text': piece.piece, 'begin': piece.begin, 'end': piece.end} for piece in spt.pieces]

