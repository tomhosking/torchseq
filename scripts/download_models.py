from transformers import BartModel, BertModel, BertTokenizer, RobertaModel, BertForQuestionAnswering, MBartModel, MBartTokenizer


mod = BartModel.from_pretrained('facebook/bart-large')
mod = RobertaModel.from_pretrained('roberta-base')

mod = BertModel.from_pretrained('bert-base-uncased')
mod = BertModel.from_pretrained('bert-base-cased')

mod = BertTokenizer.from_pretrained('bert-base-uncased')
mod = BertTokenizer.from_pretrained('bert-base-cased')

mod = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


import nltk
nltk.download('punkt', force=True)
nltk.download('wordnet', force=True)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

tokenizer = AutoTokenizer.from_pretrained("tomhosking/deberta-v3-base-debiased-nli")
model = AutoModelForSequenceClassification.from_pretrained("tomhosking/deberta-v3-base-debiased-nli")


tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-base")


mod = MBartTokenizer.from_pretrained('facebook/mbart-large-50', src_lang='en_XX', tgt_lang='en_XX')
mod = MBartTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', src_lang='en_XX', tgt_lang='en_XX')
mod = MBartModel.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')

