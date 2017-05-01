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
        self.n_words = 3 # Count SOS and EOS
      
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

class PersonaLang:
    def __init__(self, name):
        self.name = name
        self.n_persona = 1
        self.persona2index = {'UNK': 0}
        self.word2index = {'SOS': 0, 'EOS': 1, 'UNK': 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: 'UNK'}
        self.max_length = -1
        self.n_words = 3 # Count SOS and EOS
        self.persona2count = {}
        self.index2persona = {0:'UNK'}
      
    def addSentence(self, sentence):
        persona = sentence.split(':')[0]
        sentence = sentence.split(':')[1]
        # persona, sentence = sentence.split(':')
        self.addPersona(persona.strip())
        for word in sentence.strip().split(' '):
            self.addWord(word)

    def addPersona(self, persona):
        if persona not in self.persona2index:
            self.persona2index[persona] = self.n_persona
            self.persona2count[persona] = 1
            self.index2persona[self.n_persona] = persona
            self.n_persona += 1
        else:
            self.persona2count[persona] += 1

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
def normalizeString(s, persona):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.\.!?])", r"\1", s)
    s = s.lower().strip()
    s = re.sub(r"(')", r"", s)
    if persona is True:
        s = re.sub(r"[^:a-zA-Z]+", r" ", s).strip()
    else:
        s = re.sub(r"[^a-zA-Z]+", r" ", s).strip()
    return s

def prepare_data(fname, persona=False):
    # fname refers to the question answer file which is tab seperated
    
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(fname).read().strip().split('\n')
            
    lines = [normalizeString(line, persona) for line in lines]
    
    questions = lines[0::2]
    answers = lines[1::2]
    
    pairs = [(q, a) for (q, a) in zip(questions, answers)]

    if persona is True:
        lang = PersonaLang('eng')
    else:
        lang = Lang('eng')
    for line in lines:
        # print line
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

def variableFromPersona(lang, persona):
    if lang.persona2count[persona] > 20:
        indexes = [lang.persona2index[persona]]
    else:
        indexes = [lang.persona2index['UNK']]
    return Variable(cuda.LongTensor(indexes).view(-1, 1), requires_grad=False)

def variablesFromPair(lang, pair):
    input_variable = variableFromSentence(lang, pair[0])
    target_variable = variableFromSentence(lang, pair[1])
    return (input_variable, target_variable)

def variablesFromPairPersona(lang, pair):
    # create sentence pairs
    p1 = variableFromPersona(lang, pair[0].split(':')[0].strip().lower())
    p2 = variableFromPersona(lang, pair[1].split(':')[0].strip().lower())
    input_variable = variableFromSentence(lang, pair[0].split(':')[1].strip().lower())
    target_variable = variableFromSentence(lang, pair[1].split(':')[1].strip().lower())
    return (p1, input_variable, p2, target_variable)