from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import torch.nn.functional as F

class PreTrainedQA:
    def __init__(self, device=None):

        self.device = torch.device('cuda') if device is None else device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad').to(self.device)

        # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # self.model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
        
        
        


    def infer_single(self, question, text):
        input_ids = self.tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.model(torch.tensor([input_ids]).to(self.device), token_type_ids=torch.tensor([token_type_ids]).to(self.device))
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        return ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])


    def infer_batch(self, question_list, text_list):
        
        input_ids_batch = [self.tokenizer.encode(question, text) for question, text in zip(question_list, text_list)]
        token_type_ids_batch = [[0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))] for input_ids in input_ids_batch]
        start_scores, end_scores = self.model(self.pad_batch(input_ids_batch), token_type_ids=self.pad_batch(token_type_ids_batch))
        all_tokens_batch = [self.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in input_ids_batch]

        return [self.extract_answer(all_tokens_batch[ix], start_scores[ix], end_scores[ix], token_type_ids_batch[ix]) for ix in range(len(start_scores))]

    def extract_answer(self, tokens, start_scores, end_scores, type_ids):
        # If the answer is actually in the question... then this has failed!!
        if torch.argmax(end_scores) <= type_ids.index(1)-1:
            return ""
        return ' '.join(tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).replace(' ##', '')

    def pad_batch(self, batch):
        pad_id = self.tokenizer.pad_token_id
        max_len = max([len(x) for x in batch])
        padded_batch = [F.pad(torch.tensor(x), (0, max_len-len(x)), value=pad_id) for x in batch]
        return torch.stack(padded_batch, 0).to(self.device)



if __name__ == "__main__":
    instance = PreTrainedQA()

    print(instance.infer_batch(["Who was Jim Henson?", "Who was a nice puppet?"], ["Jim Henson was a nice puppet", "Jim Henson was a nice puppet"]))