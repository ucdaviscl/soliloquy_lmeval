# soliloquy_lmeval
Tools for creation of augmented language models for speech recognition in specific domains with small utterance sets.

These script are used to generate sentence alternatives (i.e., paraphrases) using an FST to generate proposed alternative sentences. The sentences are then scored using a pretrained GPT-2 model, allowing us to select the top *n* sentences. Finally we add the top *n* sentences to our original training data to create augemented training data for our language models.

## Generating alternate sentences:
Use get_fst_paraphrases.py to generate alternate sentences using an FST to identify likely alternates.

```
python get_fst_paraphrases.py -v pretrained_word_vectors.txt -f large_FST_language_model/
  -i input_sentences -o output_txt_file -n 50 -p output_pickle_file
```

The output text file contains the proposed alternatives. Similarly, the output pickle file contains the proposed alternatives in a python list which can easily be used downstream in the pipeline.

## Creating augmented training data:
Use augment_training.py to rescore the proposed alternate sentences using GPT2, then create augmented training sets, adding from 1 to 5 alternates for each training sentence. The result will be five new augmented training files, with each file augmented by a different number of alternates per input sentence.

```
python augment_training.py -p train_alternatives.pickle /
  -t training_file.txt -o output_prefix
```
Output files will begin with the output prefix follwed by the number of sentences added for each input sentence. If the prefix is "train_augmented_" output will look like *train_augmented_5.txt*.

## Evaluating resulting training data
The final step is to generate language models using the augmented training data to determine how augmenting the data affects perplexity, artificial word error rate, and vocabulary coverage.

First, all datasets should be word tokenized (this can be done using the tokenizer.py script in the variation folder) and lowercased. We also insert a period at the end of the sentence when punctuation is missing.

##### Perplexity
To build language models using KenLM and calculate LM perplexity given a test set run lm_evaluation.py. This script will call KenLM to build an LM using the specified training file with the given ngram order. It will then calculate perplexity on the given test file. This script can also be used to evaluate an existing ARPA language model.
```
python lm_evaluation.py -t test_file.txt -r training_file.txt -o ngram_order
```
Output of this script will be an LM saved in *models/my_model.arpa*. Rerunning the script will overwrite this model, so if you need to save your model, you should rename it prior to rerunning the script.

##### Artificial Word Error Rate
To calculate artificial word error rate for an LM, the ARPA LM file output by KenLM should be converted to a binary FST. This can be done with OpenFST's OpenGRM package. 
```
ngramread --ARPA --epsilon_symbol='<eps>' input.arpa > output.fst
```
You can then run the awer.py script to calculate AWER. We provide a pretrained unigram model, tr.unigrams, trained on Wikipedia data, though you can create your own unigram model with the unigrams.py script.
```
awer.py -u unigram_model -f fst_file.fst < test_data
```
If you would like you AWER results written to file, you my uncomment the outfile lines (145 & 162) in the awer.py script.

##### Vocabulary Coverage
To calculate the vocabulary coverage of your test file given your training file, you may run oov.py. This script will tell you the number of types in your training file, and how many types in your test file are not in the training vocabulary.

```
python oov.py training_file.txt test_file.txt
```

