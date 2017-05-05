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

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
MAX_LENGTH = 20

NO_DECODER = 0
DECODER_INPUT_TYPE = 1
DECODER_TARGET_TYPE = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'PAD':PAD_token, 'SOS': SOS_token, 'EOS': EOS_token, 'UNK': UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token:'PAD', SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.max_length = -1
        self.n_words = 4 # Count SOS and EOS
      
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

def indexesFromSentence(lang, sentence, decoder_flag):
#     return [lang.word2index[word] for word in sentence.split(' ')]
    
    indices = []
    if decoder_flag == DECODER_INPUT_TYPE:
        indices.append(SOS_token)

    words = sentence.split()
    sentence_length = len(words)
    for i in xrange(lang.max_length):
        if i < sentence_length:
            word = words[i]
            if lang.word2count[word] <= 20:
                word = 'UNK' # replace rare words with UNK
        else:
            word = 'PAD'
        index = lang.word2index[word]
        indices.append(index)
    if decoder_flag == NO_DECODER:
        if sentence_length >= lang.max_length:
            indices[lang.max_length - 1] = EOS_token
        elif sentence_length < lang.max_length:
            indices[sentence_length] = EOS_token
    elif decoder_flag == DECODER_TARGET_TYPE:
        if sentence_length >= lang.max_length:
            indices.append(EOS_token)
        elif sentence_length < lang.max_length:
            indices[sentence_length] = EOS_token
            indices.append(PAD_token)

    return indices, min(sentence_length, lang.max_length - 1)

def variableFromSentence(lang, sentence, decoder_flag=NO_DECODER):
    indexes, length = indexesFromSentence(lang, sentence, decoder_flag)
    return (Variable(cuda.LongTensor(indexes).view(-1, 1), requires_grad=False), length)

def variableFromPersona(lang, persona):
    if lang.persona2count[persona] > 20:
        indexes = [lang.persona2index[persona]]
    else:
        indexes = [lang.persona2index['UNK']]
    return Variable(cuda.LongTensor(indexes).view(-1, 1), requires_grad=False)

def variablesFromPair(lang, pair):
    encoder_input, encoder_sequence_length = variableFromSentence(lang, pair[0])
    decoder_input, decoder_sequence_length = variableFromSentence(lang, pair[1], decoder_flag=DECODER_INPUT_TYPE)
    decoder_target, _ = variableFromSentence(lang, pair[1], decoder_flag=DECODER_TARGET_TYPE)
    return (encoder_input, encoder_sequence_length, decoder_input, decoder_target, decoder_sequence_length)

def variablesFromPairPersona(lang, pair):
    # create sentence pairs
    p1 = variableFromPersona(lang, pair[0].split(':')[0].strip().lower())
    p2 = variableFromPersona(lang, pair[1].split(':')[0].strip().lower())
    input_variable = variableFromSentence(lang, pair[0].split(':')[1].strip().lower())
    target_variable = variableFromSentence(lang, pair[1].split(':')[1].strip().lower())
    return (p1, input_variable, p2, target_variable)