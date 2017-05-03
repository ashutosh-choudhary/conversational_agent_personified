import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import utils
import numpy as np
# import tensorflow as tf

class EncoderRNN(nn.Module):
    def __init__(self, lang, hidden_size, max_length, emb_dims):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.emb_dims = emb_dims
        self.embedding = nn.Embedding(lang.n_words, emb_dims).cuda()
        self.rnn = nn.LSTM(emb_dims, hidden_size, batch_first=True).cuda()
        
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = embedded.view(input.size()[0], self.max_length, -1)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return (Variable(cuda.FloatTensor(1, batch_size, self.hidden_size).zero_()),
               Variable(cuda.FloatTensor(1, batch_size, self.hidden_size).zero_()))


class DecoderRNN(nn.Module):
    def __init__(self, lang, hidden_size, context_size, persona_size, emb_dims, max_length, embedding):
        super(DecoderRNN, self).__init__()
        self.emb_dims = emb_dims
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.max_length = max_length
        self.embedding = embedding
        if persona_size:
            self.rnn = nn.LSTM(emb_dims + context_size + persona_size, hidden_size, batch_first=True).cuda()
        else:
            self.rnn = nn.LSTM(emb_dims + context_size, hidden_size, batch_first=True).cuda()
        self.out = nn.Linear(hidden_size, lang.n_words).cuda()
        self.softmax = nn.LogSoftmax()
        
    def forward(self, input, hidden, context, p1, p2):
        output = self.embedding(input)
        # output is N x T x D, need to concatenate with context which is N x H to produce N x T x (D + H)
        items = [output, context.view(self.context_size)]
        if p1 is not None and p2 is not None:
            # Only speaker embedding for now
            items.append(p2.view(1, -1))
        output = torch.cat(items, 1)#.view(1, 1, -1)
        print output.size()

        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        return (Variable(cuda.FloatTensor(1, batch_size, self.hidden_size).zero_()),
               Variable(cuda.FloatTensor(1, batch_size, self.hidden_size).zero_()))

class AttentionDecoder(nn.Module):
    def __init__(self, lang, max_length, hidden_size, context_size, persona_size, D_size, emb_dims, embedding):
        super(AttentionDecoder, self).__init__()
        self.emb_dims = emb_dims
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding = embedding
        if persona_size:
            self.rnn = nn.LSTM(emb_dims + context_size + persona_size, hidden_size, batch_first=True).cuda()
        else:
            self.rnn = nn.LSTM(emb_dims + context_size, hidden_size, batch_first=True).cuda()
        self.out = nn.Linear(hidden_size, lang.n_words).cuda()
        self.softmax = nn.LogSoftmax()

        self.a_layer = torch.nn.Softmax()
        self.r_layer = torch.nn.Linear(hidden_size, D_size).cuda() # here second context size is D
        self.u_layer = torch.nn.Linear(D_size, 1).cuda() # here context size is D
        
    def forward(self, input, hidden, encoder_states, wf_mat, p1, p2):
        
        output = self.embedding(input).view(1, -1)

        r_t = torch.t(self.r_layer(hidden[0][0])) # D x 1 get the hidden state of the first element in the batch
        # print "r", r_t.size()
        # print "w_f", wf_mat.size()
        tanh = F.tanh(wf_mat.add(r_t.expand_as(wf_mat))) # D x f
        # print "tanh", tanh.size()
        u_t = self.u_layer(torch.t(tanh)) # f x 1
        # print "u", u_t.size()
        a_t = self.a_layer(u_t) # f x 1
        # print "a:", a_t.size()
        context = torch.mm(torch.t(a_t), encoder_states) # 1 x H
        # print "context", context.size()

        items = [output, context.view(1, -1)]

        if p1 is not None and p2 is not None:
            # Only speaker embedding for now
            items.append(p2.view(1, -1))

        output = torch.cat(items, 1).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        # set the F matrix to None
        return (Variable(cuda.FloatTensor(1, 1, self.hidden_size).zero_()),
               Variable(cuda.FloatTensor(1, 1, self.hidden_size).zero_()))

class Seq2Seq(object):
    # Does not work on batches yet, just works on a single question and answer

    def __init__(self, lang, enc_size, dec_size, emb_dims, max_length, learning_rate, attention=False, reload_model=False, persona=False, persona_size=None):

        self.attention = attention
        self.persona = persona
        if persona is True:
            self.persona_embedding = nn.Embedding(lang.n_persona, persona_size).cuda() # emb_dims of character is 20
        if reload_model is True:
            self.encoder = torch.load(open('../models/encoder.pth'))
            self.decoder = torch.load(open('../models/decoder.pth'))        
        else:
            self.encoder = EncoderRNN(lang, enc_size, max_length, emb_dims)

            if attention is True:
                self.D_size = self.encoder.hidden_size
                self.decoder = AttentionDecoder(lang, max_length, dec_size, enc_size, persona_size, self.D_size, emb_dims, self.encoder.embedding)
            else:
                self.decoder = DecoderRNN(lang, dec_size, enc_size, emb_dims, persona_size, max_length, self.encoder.embedding)

        self.max_length = max_length
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)
        self.lang = lang
        if attention is True:
            
            self.wf_layer = torch.nn.Linear(self.encoder.hidden_size, self.D_size).cuda()
        
        self.criterion = nn.NLLLoss() # Negative log loss
        # self.summary_op = tf.summary.merge_all()

    def save_model(self):

        torch.save(self.encoder, '../models/encoder.pth')
        torch.save(self.decoder, '../models/decoder.pth')

    def forward(self, batch_pairs, train=True):

        N = len(batch_pairs)

        # pair = tuple of (question, answer)
        if self.persona is True:
            (persona1, input_variable, input_length, persona2, target_variable, target_length) = utils.variablesFromPairPersona(self.lang, pair)
            p1 = self.persona_embedding(persona1).view(1, -1)
            p2 = self.persona_embedding(persona2).view(1, -1)
        else:
            input_batch = Variable(cuda.LongTensor(N, self.max_length).zero_(), requires_grad=False)
            target_batch = Variable(cuda.LongTensor(N, self.max_length + 1).zero_(), requires_grad=False) # start with SOS token
            input_batch_len = []
            target_batch_len = []
            for i in xrange(N):    
                (input_variable, input_length, target_variable, target_length) = utils.variablesFromPair(self.lang, batch_pairs[i])
                input_batch[i] = input_variable
                target_batch[i] = target_variable
                input_batch_len.append(input_length)
                target_batch_len.append(target_length)
            input_batch_len = cuda.LongTensor(input_batch_len)
            target_batch_len = cuda.LongTensor(target_batch_len)
            p1 = None
            p2 = None

        if train is False:
            print input_variable

        encoder_hidden = self.encoder.initHidden(N)
        decoder_hidden = self.decoder.initHidden(N)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # input_length = input_variable.size()[0]
        # target_length = target_variable.size()[0]

        loss = 0
        if self.attention is True:
            encoder_states = Variable(cuda.FloatTensor(input_length, self.encoder.hidden_size).zero_())
        
        # Encode the sentence
        # for ei in range(input_length):
        #     encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
            # if self.attention is True:
            #     encoder_states[ei] = encoder_output[0][0] # First element in batch, only hidden state and not cell state
        # print encoder_hidden[0].size(), input_batch.size()

        encoder_output, encoder_hidden = self.encoder(input_batch, encoder_hidden)

        encoder_hidden_states = Variable(cuda.FloatTensor(N, self.encoder.hidden_size).zero_())
        for i in xrange(N):
            encoder_hidden_states[i] = encoder_output[i, input_batch_len[i] - 1, :]

        # if self.attention is True:
        #     self.wf = torch.t(self.wf_layer(encoder_states)) # D x f

        # print torch.mean(encoder_output)
    	del input_variable

        
        # Decode with start symbol as SOS
        response = []
        if train is True:
            decoder_output, decoder_hidden = self.decoder(target_batch, decoder_hidden, encoder_hidden_states, p1, p2)

            assert False
            # for di in xrange(self.max_length):
            #     if self.attention is True:
            #         decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_states, self.wf, p1, p2)
            #     else:
            #         decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output[0][0], p1, p2)
            #     # TODO change the loss to batch loss considering pad symbols
            # 	if di == target_length:
            #     	break
            #     loss += self.criterion(decoder_output[0], target_variable[di])
            #     decoder_input = target_variable[di] # Teacher forcing
            #     ind = target_variable[di][0]
        else:
            # greedy decode
            response = []
            for di in xrange(self.max_length):
                if self.attention is True:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_states, self.wf, p1, p2)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output[0][0], p1, p2)
                topv, topi = decoder_output.data.topk(1)
                ind = topi[0][0]
                if ind == utils.EOS_token:
                    break
                decoder_input = Variable(cuda.LongTensor([[ind]]), requires_grad=False)
                response.append(self.lang.index2word[ind])

            # This implementation of beam search is wrong, we need to predict and follow the pointers back.
            # beam_size = 5
            # di = 0
            # while di < self.max_length:


                    
                
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
