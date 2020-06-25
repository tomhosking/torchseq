from transformers import BartModel, BertModel, BertTokenizer, RobertaModel, BertForQuestionAnswering

mod = BartModel.from_pretrained('facebook/bart-large')

mod = RobertaModel.from_pretrained('roberta-base')

mod = BertModel.from_pretrained('bert-base-uncased')
mod = BertModel.from_pretrained('bert-base-cased')

mod = BertTokenizer.from_pretrained('bert-base-uncased')
mod = BertTokenizer.from_pretrained('bert-base-cased')

mod = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")