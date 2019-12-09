import math
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import torch

class Model:
    def __init__(self):
        self.model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        self.model.eval()
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    def score(self, sentence):
        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss=self.model(tensor_input, lm_labels=tensor_input)
        ppl = math.exp(loss)
#        toks = len(tokenize_input)
#        score = math.log10(math.log(1/(ppl**toks)))
        return ppl
