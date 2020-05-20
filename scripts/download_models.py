from transformers import BartModel, BertModel, BertTokenizer, RobertaModel

mod = BartModel.from_pretrained('bart-large')

mod = RobertaModel.from_pretrained('roberta-base')

mod = BertModel.from_pretrained('bert-base-uncased')
mod = BertModel.from_pretrained('bert-base-cased')

mod = BertTokenizer.from_pretrained('bert-base-uncased')
mod = BertTokenizer.from_pretrained('bert-base-cased')