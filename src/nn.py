import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import utils
# import tensorflow as tf

class EncoderRNN(nn.Module):
    def __init__(self, lang, hidden_size, emb_dims):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb_dims = emb_dims
        self.embedding = nn.Embedding(lang.n_words, emb_dims).cuda()
        self.rnn = nn.LSTM(emb_dims, hidden_size).cuda()
        
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = embedded.view(1, 1, -1)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        return (Variable(cuda.FloatTensor(1, 1, self.hidden_size).zero_()),
               Variable(cuda.FloatTensor(1, 1, self.hidden_size).zero_()))


class DecoderRNN(nn.Module):
    def __init__(self, lang, hidden_size, context_size, emb_dims, embedding):
        super(DecoderRNN, self).__init__()
        self.emb_dims = emb_dims
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.embedding = embedding
        self.rnn = nn.LSTM(emb_dims + context_size, hidden_size).cuda()
        self.out = nn.Linear(hidden_size, lang.n_words).cuda()
        self.softmax = nn.LogSoftmax()
        
    def forward(self, input, hidden, context):
        output = self.embedding(input).view(1, -1)
        items = [output, context.view(1, -1)]
        output = torch.cat(items, 1).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return (Variable(cuda.FloatTensor(1, 1, self.hidden_size).zero_()),
               Variable(cuda.FloatTensor(1, 1, self.hidden_size).zero_()))

class Seq2Seq(object):
    # Does not work on batches yet, just works on a single question and answer

    def __init__(self, lang, enc_size, dec_size, emb_dims, max_length, learning_rate, reload_model=False):

        if reload_model:
            self.encoder = torch.load(open('../models/encoder.pth'))
            self.decoder = torch.load(open('../models/decoder.pth'))        
        else:
            self.encoder = EncoderRNN(lang, enc_size, emb_dims)
            self.decoder = DecoderRNN(lang, dec_size, enc_size, emb_dims, self.encoder.embedding)

        self.max_length = max_length
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        self.lang = lang
        self.criterion = nn.NLLLoss()
        # self.summary_op = tf.summary.merge_all()

    def forward(self, pair, train=True):

        # pair = tuple of (question, answer)

        (input_variable, target_variable) = utils.variablesFromPair(self.lang, pair)

        encoder_hidden = self.encoder.initHidden()
        decoder_hidden = self.decoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]
        
        loss = 0
        # Encode the sentence
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
    	
    	del input_variable
        decoder_input = Variable(cuda.LongTensor([[utils.SOS_token]]), requires_grad=False)

        # Decode with start symbol as SOS
        response = []
        for di in xrange(self.max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
            del decoder_input
            # TODO change the loss to batch loss considering pad symbols
            if train is True:
            	if di == target_length:
                	break
                loss += self.criterion(decoder_output[0], target_variable[di])
                decoder_input = target_variable[di] # Teacher forcing
                ind = target_variable[di][0]
            else:
                topv, topi = decoder_output.data.topk(1)
                ind = topi[0][0]
                if ind == utils.EOS_token:
                	break
                decoder_input = Variable(cuda.LongTensor([[ind]]), requires_grad=False)
            	response.append(self.lang.index2word[ind])
        
        # tf.summary.scalar('loss', loss)

        
        # Step back
        if train is True:
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
        
        del encoder_hidden
        del decoder_hidden
        del decoder_output
        del target_variable
    	response = ' '.join(response)
        return response, loss
