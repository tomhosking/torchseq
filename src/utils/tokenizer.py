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
        return BPE.instance().decode(token_id_tensor.tolist(), skip_special_tokens=True).replace(' ##', '').replace('# ', '#').replace(' ; ', ' ;')


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

        # NOTE: HF tokenizers automatically adds CLS/SEP tokens, so we have to fudge the indices to skip these
        tokenised = [{'id': token_ids[ix], 'text': token_texts[ix], 'begin': offsets[ix][0], 'end': offsets[ix][1]} for ix in range(1,len(output.tokens)-1)]
        if add_bos_eos:
            return bos + tokenised + eos
        else:
            return tokenised

