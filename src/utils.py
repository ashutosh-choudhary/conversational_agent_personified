import unicodedata
import string
import re
import random
import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LENGTH = 20

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'SOS': 0, 'EOS': 1, 'UNK': 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: 'UNK'}
        self.max_length = -1
        self.n_words = 2 # Count SOS and EOS
      
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.\.!?])", r"\1", s)
    s = s.lower().strip()
    s = re.sub(r"(')", r"", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s).strip()
    return s

def prepare_data(fname):
    # fname refers to the question answer file which is tab seperated
    
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(fname).read().strip().split('\n')
    
    lines = [normalizeString(line) for line in lines]
    
    questions = lines[0::2]
    answers = lines[1::2]
    
    pairs = [(q, a) for (q, a) in zip(questions, answers)]

    lang = Lang('eng')
    for line in lines:
        lang.addSentence(line)
    print "Done"
    return lang, pairs

def indexesFromSentence(lang, sentence):
#     return [lang.word2index[word] for word in sentence.split(' ')]
    indices = []
    for i, word in enumerate(sentence.split()):
        if i > lang.max_length:
            break
        if lang.word2count[word] <= 20:
            word = 'UNK' # replace rare words with UNK
        index = lang.word2index[word]
        indices.append(index)
    return indices

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return Variable(cuda.LongTensor(indexes).view(-1, 1), requires_grad=False)

def variablesFromPair(lang, pair):
    input_variable = variableFromSentence(lang, pair[0])
    target_variable = variableFromSentence(lang, pair[1])
    return (input_variable, target_variable)