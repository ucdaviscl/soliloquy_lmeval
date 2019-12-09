"""
This is a program to take in either:
a) a pretrained language model file (kenLM or PyTorch)
b) a indicator of model type (if using GPT-2) or
c) a training dataset

along with a test dataset file.
Test data file should be plain text with one sentence per line.

if you provide

"""
import sys, io, nltk
import kenlm
#import torch
#import inquirer
import subprocess
import numpy as np
import copy, math, argparse
#import gpt_score
#from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
#from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

gpt = True

try:
    import gpt_score
except ImportError:
    gpt = False

parser = argparse.ArgumentParser(description = "Language Model Evaluation")
parser.add_argument('-m', '--model', type = str, default = '', help = 'language model', required = False)
parser.add_argument('-t', '--test_file', type = str, default = '', help = 'test text file', required = True)
parser.add_argument('-r', '--train_file', type = str, default = '', help = 'train text file', required = False)
parser.add_argument('-o', '--ngram_order', type = int, default = 2, help = 'ngram order', required = False)

# A class for loading a LM and training if necessary
class LM:
    def __init__(self, model_file, train_file, order):
        if ".arpa" in model_file or ".binary" in model_file:
            self.model_type = "kenlm"
        elif "gpt2" in model_file:
            self.model_type = "gpt2"
        elif ".pt" in model_file[:-3]:
            self.model_type = "torch"
        elif model_file == "":
            if train_file == "":
                print("Error: please provide a valid train file or pretrained model")
#                exit()
            else:
                self.model_type = "train_me_plz"
        self.order = order
        self.model_file = model_file
        self.train_file = train_file
        self.model = self.load_lm()

    def load_lm(self):
        if self.model_type == "kenlm":
            lm = kenlm.Model(self.model_file)
            return lm
        elif self.model_type == "train_me_plz":
#            self.process_train_data(self.train_file)
            train_input = self.train_file
            subprocess.call("../kenlm/bin/lmplz -o %d < %s > models/my_model.arpa" %(self.order, train_input), shell=True)
            lm = kenlm.Model("models/my_model.arpa")
            self.model_type = "kenlm"
            return lm
        elif self.model_type == "gpt2":
            lm = gpt_score.Model()
            return lm
            
    #Return the log10 probability of the sentence given the LM
    def score(self, sent):
        prob = self.model.score(sent)
        return prob

    def process_train_data(self, train_file):
        outfile = io.open("data/my_train_data.txt", encoding="utf-8", mode="w+")
        with io.open(train_file, encoding="utf-8") as infile:
            for line in infile:
                tokens = [tok.lower() for tok in nltk.word_tokenize(line)]
                outfile.write(" ".join(tokens).strip() + '\n')



def process_test_data(test_file):
    sents = []
    for line in io.open(test_file, encoding="utf-8"):
#        tokens = [tok.lower() for tok in nltk.word_tokenize(line)]
#        sents.append(" ".join(tokens))
        sents.append(line.strip())
    return sents


def calc_perplexity(test_data, lm):
    tot_score = 0
    tokens = 0

    for sent in test_data:
#        print(sent)
#        print(math.exp(lm.score(sent)))
        prob = lm.score(sent)/math.log10(2)
        tot_score += prob
        toks = len(sent.split()) + 1
        tokens += toks

    ppl = 2**((-1/tokens)*tot_score)
    return ppl


def main():
    args = parser.parse_args()
    model = args.model
    test_file = args.test_file
    train_file = args.train_file
    order = args.ngram_order

    lm = LM(model, train_file, order)

#    lm, tokenizer = load_lm(model_type, model, train_file)
#    test_data = process_test_data(test_file)
    test_data = io.open(test_file, encoding="utf-8").readlines()
    ppl = calc_perplexity(test_data, lm)

    print("Model perplexity on test data: ", ppl)

main()
