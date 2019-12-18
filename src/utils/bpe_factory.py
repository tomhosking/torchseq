import re
from bpemb import BPEmb
from utils.sentencepiece_pb2 import SentencePieceText

import unicodedata


class BPE_bpemb:
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


from transformers import  BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel

class BPE:
    _instance = None

    pad_id = None
    embedding_dim = None
    bos_id = None
    eos_id = None

    @staticmethod
    def decode(token_id_tensor):
        return BPE.instance().decode(token_id_tensor.tolist(), skip_special_tokens=True)


    @staticmethod
    def instance():
        if BPE._instance is None:
            BPE._instance = BertTokenizer.from_pretrained('bert-base-uncased')
            # BPE._instance = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            BPE.pad_id = BPE._instance.pad_token_id

            # special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]'}

            BPE.bos_id = BPE._instance.cls_token_id
            BPE.eos_id = BPE._instance.sep_token_id

            # num_added_toks = BPE._instance.add_special_tokens(special_tokens_dict)
            
            
            model = BertModel.from_pretrained('bert-base-uncased')
            # model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            BPE._instance.embeddings = model.embeddings.word_embeddings.weight.data

            del model
            # print(len(BPE._instance))
            # exit()
        return BPE._instance

    @staticmethod
    def tokenise(text, add_bos_eos=False):
        tokens = BPE.instance().tokenize(text)

        # clean_text = BPE.instance().basic_tokenizer._clean_text(text)
        # clean_text = BPE.instance().basic_tokenizer._run_strip_accents(clean_text).lower()

        clean_text, offsets = normalise(text)

        pieces = []
        offset = 0
        for tok in tokens:
            if tok == BPE.instance().unk_token:
                continue
            needle = tok[2:] if tok[:2] == '##' else tok

            new_offset = clean_text.find(needle, offset)
            

            if new_offset < 0:
                print('Couldnt find token: {:} (pos: {:})'.format(needle, offset))
                print(text)
                print(clean_text)
                exit()
            offset = new_offset
            # char_offsets.append( (offset, offset+len(tok)) )

            # Now convert the offset in clean_text to the original...

            for x in offsets:
                if x[1] == offset:
                    orig_offset = x[0]
            
            
            pieces.append(
                {
                    'id': BPE.instance().convert_tokens_to_ids(tok),
                    'text': tok,
                    'begin_new': offset,
                    'begin': get_byte_offsets(clean_text, orig_offset),
                    'end': get_byte_offsets(clean_text, orig_offset + len(needle))
                }
            )
            # print(tok, offset, offset + len(needle))
            offset = offset + len(needle) - 1

        # if 'is usually translated into english as "virtuous behavior"' in text.lower():
        #     print(text)
        #     print(clean_text)
        #     print(offsets)
        #     print(len(text), len(clean_text))
        #     print(pieces)
        #     exit()

        # if 'Persia' in text:
        #     print(text)
        #     print(tokens)
        #     print(pieces)
            
        #     exit()

        # bos = [{'id': BPE.bos_id, 'text': BPE.instance().bos_token, 'begin': 0, 'end': 0}]
        # eos = [{'id': BPE.eos_id, 'text': BPE.instance().eos_token, 'begin': len(text), 'end': len(text)}]

        bos = [{'id': BPE.instance().cls_token_id, 'text': BPE.instance().cls_token, 'begin': 0, 'end': 0}]
        eos = [{'id': BPE.instance().sep_token_id, 'text': BPE.instance().sep_token, 'begin': len(text), 'end': len(text)}]

        tokenised = [{'id': piece['id'], 'text': piece['text'], 'begin': piece['begin'], 'end': piece['end']} for piece in pieces]
        if add_bos_eos:
            return bos + tokenised + eos
        else:
            return tokenised


# This fn converts character positions to byte positions - why? Because SentencePiece works in terms of byte offsets, and SQuAD contains some full width characters!
def get_byte_offsets(text, character_offset):
    return character_offset
    # needs_conversion = len(text) != len(text.encode('utf8'))
    # if needs_conversion:
    #     begin_offset = len(text[:character_offset].encode('utf8'))
    # else:
    #     begin_offset = character_offset
    # return begin_offset

def _run_strip_accents(text):
    # nfkd_form = unicodedata.normalize('NFKD', text)
    # return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    output = []
    offsets=[]
    delta=0
    for i, char in enumerate(text):
        char = unicodedata.normalize("NFD", char) # Normalise a char at a time as otherwise it's not a length preserving process!!
        if len(char) > 1:# and unicodedata.category(char[1]) == 'Mn':
#             print(char, char[0], [unicodedata.category(c) for c in char])
#             char = char[0]
            if unicodedata.category(char[1]) == 'Mn':
                output.append(char[0])
            else:
                output.append(char)
        else:
            cat = unicodedata.category(char)
            # print(char, cat)
            if cat == "Mn":
                # output.append('_')
                delta -= 1
            else:
                output.append(char)
        offsets.append((i, i+delta))
    return "".join(output), offsets


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

def _clean_text(text,prev_offsets=None):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    offsets = []
    delta = 0
    j=0
    for i,char in enumerate(text):
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            delta -= 1
        elif _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
        
        # Move the cursor
        while j < len(prev_offsets)-1 and prev_offsets[j+1][1] <= i:
            j += 1
        
        offsets.append((prev_offsets[j][0], i+delta))
    return "".join(output), offsets

def normalise(text):
    # return text

    clean_text, offsets1 = _run_strip_accents(text)

    clean_text, offsets2 = _clean_text(clean_text, offsets1)

    

    # Combine offsets
    # if 'is usually translated into english as "virtuous behavior"' in text.lower():
    #     print(offsets1)
    #     print(offsets2)
    #     exit()

    return clean_text.lower(), offsets2