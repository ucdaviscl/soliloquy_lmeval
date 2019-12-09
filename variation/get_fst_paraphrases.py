import sys
import argparse
import pickle
import sentalter
import tokenizer

parser = argparse.ArgumentParser(description='Generate paraphrases with fst')
parser.add_argument('-v', '--vectors', type = str, default = '', help = 'word vectors', required = True)
parser.add_argument('-f', '--fst_lm', type = str, default = '', help = 'fst language model', required = True)
parser.add_argument('-d', '--onmt_dir', type = str, default = '', help = 'OpenNMT intallation directory')
parser.add_argument('-m', '--onmt_lm', type = str, default = '', help = 'OpenNMT language model')
parser.add_argument('-k', '--kenlm', type = str, default = '', help = 'KenLM language model')
parser.add_argument('-i', '--input', type = str, help = 'text file input', required = True)
parser.add_argument('-o', '--output', type = str, help = 'text file output')
parser.add_argument('-p', '--pickle', type = str, help = 'pickle file output')
parser.add_argument('-n', '--num', type = int, default = 5, help = 'number of paraphrases to generate for each sentence')
params = parser.parse_args()

output_dict = {}

print('Initializing...')
lv = sentalter.AlterSent(params.vectors, params.fst_lm, params.onmt_dir, params.onmt_lm, params.kenlm, 50000)

if params.output:
	f = open(params.output, 'w')
	myprint = lambda x: f.write(str(x)+'\n')
else:
	f = None
	myprint = print
eprint = lambda x: print(x, file = sys.stderr)

with open(params.input) as fin:
	sents = dict()
	for line in fin:
		l = line.rstrip('\n ')
		if l not in sents:
			sents[l] = 0
		sents[l]+=1

i=1
for sent in sents:
	eprint('Sentence {} of {}'.format(i, len(sents)))
	i+=1
	words = tokenizer.word_tokenize(sent)
	lines = lv.fst_alter_sent(words, params.num)
#	print(lines)
#	for j in range(sents[sent]):
	if len(lines) >=10:
		lines = lines[:10]
	for x in lines:
		myprint(x[2])
	output_dict[sent] = lines

eprint('Output file should be randomly shuffled before used.')

if params.pickle:
	p_out = open(params.pickle, 'wb')
	pickle.dump(output_dict, p_out)

if f:
	f.close()
