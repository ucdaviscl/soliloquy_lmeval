import gpt_score
import pickle
import sys, random, argparse

parser = argparse.ArgumentParser(description = "Adding alternatives to test file")
parser.add_argument('-p', '--pickle_file', type = str, default = '', help = 'pickle file of alternatives', required = True)
parser.add_argument('-t', '--train_file', type = str, default = '', help = 'train text file', required = True)
#parser.add_argument('-n', '--num_add', type = int, default = 1, help = 'number to add', required = False)
parser.add_argument('-o', '--out_prefix', type = str, default = '', help = 'output train file', required = True)

args = parser.parse_args()

sent_dict = pickle.load(open(args.pickle_file, mode="rb"))
train_file = open(args.train_file)
model = gpt_score.Model()

sents_to_add_1 = []
sents_to_add_2 = []
sents_to_add_3 = []
sents_to_add_4 = []
sents_to_add_5 = []

index = 0

print("Number of sents to process: ", len(sent_dict), flush=True)

for sent in sent_dict:
    scored_alternatives = []

    for alternative in sent_dict[sent]:
        alt_sent = alternative[2]
        score = model.score(alt_sent)
        scored_alternatives.append((alt_sent, score))

    scored_alternatives.sort(key = lambda x: x[1])
#    for i in range(args.num_add):
#            sents_to_add.append(scored_alternatives[i][0])
    sents_to_add_1.append(scored_alternatives[0][0])

    for i in range(2):
        if len(scored_alternatives) > i:
            sents_to_add_2.append(scored_alternatives[i][0])
        else:
            continue
    for i in range(3):
        if len(scored_alternatives) > i:
            sents_to_add_3.append(scored_alternatives[i][0])
        else:
            continue
    for i in range(4):
        if len(scored_alternatives) > i:
            sents_to_add_4.append(scored_alternatives[i][0])
        else:
            continue
    for i in range(5):
        if len(scored_alternatives) > i:
            sents_to_add_5.append(scored_alternatives[i][0])
        else:
            continue

    index += 1
    print('Completed sentence: '+ str(index), flush=True)

train_lines = train_file.readlines()

new_train_1 = train_lines + sents_to_add_1
random.shuffle(new_train_1)
new_train_2 = train_lines + sents_to_add_2
random.shuffle(new_train_2)
new_train_3 = train_lines + sents_to_add_3
random.shuffle(new_train_3)
new_train_4 = train_lines + sents_to_add_4
random.shuffle(new_train_4)
new_train_5 = train_lines + sents_to_add_5
random.shuffle(new_train_5)

outfile_1 = open(args.out_prefix + "1.txt", mode="w+")
outfile_2 = open(args.out_prefix + "2.txt", mode="w+")
outfile_3 = open(args.out_prefix + "3.txt", mode="w+")
outfile_4 = open(args.out_prefix + "4.txt", mode="w+")
outfile_5 = open(args.out_prefix + "5.txt", mode="w+")

for line in new_train_1:
    outfile_1.write(line.strip() + '\n')
for line in new_train_2:
    outfile_2.write(line.strip() + '\n')
for line in new_train_3:
    outfile_3.write(line.strip() + '\n')
for line in new_train_4:
    outfile_4.write(line.strip() + '\n')
for line in new_train_5:
    outfile_5.write(line.strip() + '\n')

outfile_1.close()
outfile_2.close()
outfile_3.close()
outfile_4.close()
outfile_5.close()
