import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from pytorch_rnn import *
import utils
import numpy as np
from scipy.misc import logsumexp
from pynvml import *
# import tensorflow as tf
nvmlInit()
GPU_handle = nvmlDeviceGetHandleByIndex(0)

def print_gpu_status(label=-1):
    gpu_info = nvmlDeviceGetMemoryInfo(GPU_handle)
    print "line:", label, "MEM:", float(gpu_info.free) / (1024**3)

class EncoderRNN(nn.Module):
    def __init__(self, lang, hidden_size, max_length, emb_dims):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.emb_dims = emb_dims
        self.embedding = nn.Embedding(lang.n_words, emb_dims).cuda()
        self.rnn = LSTM(emb_dims, hidden_size, batch_first=True).cuda()
        
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
        self.lang = lang
        if persona_size:
            self.rnn = LSTM(emb_dims + context_size + persona_size, hidden_size, batch_first=True).cuda()
        else:
            self.rnn = LSTM(emb_dims + context_size, hidden_size, batch_first=True).cuda()
        self.out = nn.Linear(hidden_size, lang.n_words).cuda()
        self.softmax = nn.LogSoftmax()
        
    def forward(self, input, hidden, context, question_persona_batch, answer_persona_batch):

        # context = N x H need to convert N x (T + 1) x H
        # input = N x (T + 1)
        # hidden = decoder_hidden

        N, T = input.size()
        T -= 1 # as input is T+1
        H = context.size()[1]

        output = self.embedding(input)

        multi_context = [context for t in xrange(T + 1)]

        multi_context = torch.cat(multi_context, 1).view(N, T+1, H)
        # output is N x (T + 1) x D, need to concatenate with context which is N x H to produce N x T x (D + H)
        items = [output, multi_context]
        if question_persona_batch is not None and answer_persona_batch is not None:
            persona_size = answer_persona_batch.size()[1]
            # Only speaker embedding for now
            p2 = [answer_persona_batch.view(N, 1, persona_size) for t in xrange(T+1)]
            p2 = torch.cat(p2, 2).view(N, T+1, persona_size)
            items.append(p2)

        output = torch.cat(items, 2).view(N, T+1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = output.contiguous().view(-1, H)
        output = self.softmax(self.out(output)).view(N, T+1, self.lang.n_words)
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
        self.lang = lang
        self.context_size = context_size
        if persona_size:
            self.rnn = LSTM(emb_dims + context_size + persona_size, hidden_size, batch_first=True).cuda()
        else:
            self.rnn = LSTM(emb_dims + context_size, hidden_size, batch_first=True).cuda()
        self.out = nn.Linear(hidden_size, lang.n_words).cuda()
        self.softmax = nn.LogSoftmax()
        self.D_size = D_size
        self.a_layer = torch.nn.Softmax()
        self.r_layer = torch.nn.Linear(hidden_size, D_size).cuda() # here second context size is D
        self.u_layer = torch.nn.Linear(D_size, 1).cuda() # here context size is D
        
    def forward(self, input, hidden, encoder_states, wf_mat, mask, question_persona_batch, answer_persona_batch):
        
        N = input.size()[0]
        T = encoder_states.size()[1]

        output = self.embedding(input).view(N, -1)

        r_t = self.r_layer(hidden[0][0]) # D x 1 get the hidden state of the first element in the batch
        # print "r", r_t.size()
        # print "w_f", wf_mat.size()

        r_copy_t = [r_t.view(N, 1, self.D_size) for t in xrange(T)]
        r_copy_t = torch.cat(r_copy_t, 1)

        tanh = F.tanh(wf_mat + r_copy_t) # N x T x D
        # NT x D
        # pass to u layer
        # reshape to N x T x 1
        tanh = tanh.contiguous().view(-1, self.D_size)

        # print "tanh", tanh.size()
        u_t = self.u_layer(tanh).view(N, T, 1) # f x 1
        temp_zero = Variable(cuda.FloatTensor(N, T).zero_(), requires_grad=False)
        u_t = torch.addcmul(temp_zero,u_t, mask).view(N, T)

        del temp_zero

        # print "u", u_t.size()
        a_t = self.a_layer(u_t) # N x T

        # reshape encoder_states from N x T x H to NT x H
        encoder_states = encoder_states.contiguous().view(N*T, self.context_size)
        # reshape a_t to NTx1
        a_t = a_t.contiguous().view(N*T, 1)

        a_t = [a_t for h in xrange(self.context_size)]
        a_t = torch.cat(a_t, 1)

        # weighted product
        temp_zero = Variable(cuda.FloatTensor(N*T, self.context_size).zero_())
        context = torch.addcmul(temp_zero, a_t, encoder_states).view(N, T, self.context_size) # this is NT x H , reshape to N x T x H
        del temp_zero
        # get weighted sum along the time axis
        context = torch.sum(context, 1).view(N, self.context_size) # context size should be N x H

        items = [output, context]
        if question_persona_batch is not None and answer_persona_batch is not None:
            # Only speaker embedding for now
            items.append(answer_persona_batch)

        output = torch.cat(items, 1)
        output = F.relu(output).view(N, 1, -1)

        output, hidden = self.rnn(output, hidden)  # output will be N x T x H

        # reshape output to N x H, convert to softmax to get N x V, reshape to N x 1 x V
        output = output.contiguous().view(N, self.hidden_size)
        output = self.softmax(self.out(output)).view(N, 1, self.lang.n_words)

        return output, hidden

    def initHidden(self, batch_size):
        return (Variable(cuda.FloatTensor(1, batch_size, self.hidden_size).zero_()),
               Variable(cuda.FloatTensor(1, batch_size, self.hidden_size).zero_()))

class Seq2Seq(object):
    # Does not work on batches yet, just works on a single question and answer

    def __init__(self, lang, enc_size, dec_size, emb_dims, max_length, learning_rate, #graph,
            attention=False, reload_model=False, persona=False, persona_size=None):

        self.attention = attention
        self.persona = persona
        self.persona_size = None

        # self.graph = graph
        if persona is True:
            self.persona_size = persona_size
            self.persona_embedding = nn.Embedding(lang.n_persona, persona_size).cuda() # emb_dims of character is 20
        if reload_model is True:
            self.encoder = torch.load(open('../models/encoder.pth'))
            self.decoder = torch.load(open('../models/decoder.pth'))        
        else:
            self.encoder = EncoderRNN(lang, enc_size, max_length, emb_dims)
            if attention is True:
                self.D_size = self.encoder.hidden_size
                self.decoder = AttentionDecoder(lang, max_length, dec_size, enc_size, self.persona_size, self.D_size, emb_dims, self.encoder.embedding)
            else:
                self.decoder = DecoderRNN(lang, dec_size, enc_size, self.persona_size, emb_dims, max_length, self.encoder.embedding)

        self.max_length = max_length
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)
        self.lang = lang
        if attention is True:
            self.wf_layer = torch.nn.Linear(self.encoder.hidden_size, self.D_size).cuda()
        
        self.criterion = nn.NLLLoss() # Negative log loss
        # with self.graph.as_default() as graph:
        #     self.writer = tf.summary.FileWriter('logs/')
        #     self.summary_op = tf.summary.merge_all()

    def save_model(self, model_path='../models/seq2seq/'):
        torch.save(self.encoder.state_dict(), open(model_path + 'encoder', 'wb'))
        torch.save(self.decoder.state_dict(), open(model_path + 'decoder', 'wb'))
        if self.attention is True:
            torch.save(self.wf_layer.state_dict(), open(model_path + 'wf', 'wb'))
        if self.persona is True:
            torch.save(self.persona_embedding.state_dict(), open(model_path + 'persona', 'wb'))

    def load_model(self, model_path='../models/seq2seq/'):
        enc_state = torch.load(open(model_path + 'encoder'))
        self.encoder.load_state_dict(enc_state)
        dec_state = torch.load(open(model_path + 'decoder'))
        self.decoder.load_state_dict(dec_state)
        if self.attention is True:
            wf_state = torch.load(open(model_path + 'wf'))
            self.wf_layer.load_state_dict(wf_state)
        if self.persona is True:
            pers_state = torch.load(open(model_path + 'persona'))
            self.persona_embedding.load_state_dict(pers_state)

    def forward(self, batch_pairs, train=True, decode_flag='greedy', random_sample_size=100, beam_size=2):

        N = len(batch_pairs)
        encoder_input_batch = Variable(cuda.LongTensor(N, self.max_length).zero_(), requires_grad=False)
        decoder_input_batch = Variable(cuda.LongTensor(N, self.max_length + 1).zero_(), requires_grad=False) # start with SOS token
        decoder_target_batch = Variable(cuda.LongTensor(N, self.max_length + 1).zero_(), requires_grad=False)
        if self.persona is True:
            question_persona_batch = Variable(cuda.FloatTensor(N, self.persona_size).zero_(), requires_grad=False)
            answer_persona_batch = Variable(cuda.FloatTensor(N, self.persona_size).zero_(), requires_grad=False)
        else:
            question_persona_batch = None
            answer_persona_batch = None
        encoder_input_batch_len = []
        decoder_input_batch_len = []
        for i in xrange(N):    
            if self.persona is False:
                (encoder_input, encoder_sequence_length, decoder_input, decoder_target, decoder_sequence_length) = utils.variablesFromPair(self.lang, batch_pairs[i])
            else:
                (p1, encoder_input, encoder_sequence_length, p2, decoder_input, decoder_target, decoder_sequence_length) = utils.variablesFromPairPersona(self.lang, batch_pairs[i])    
                question_persona_batch[i] = self.persona_embedding(p1).view(1, -1)
                answer_persona_batch[i] = self.persona_embedding(p2).view(1, -1)
            encoder_input_batch[i] = encoder_input
            decoder_input_batch[i] = decoder_input
            decoder_target_batch[i] = decoder_target
            encoder_input_batch_len.append(encoder_sequence_length)
            decoder_input_batch_len.append(decoder_sequence_length)
        encoder_input_batch_len = cuda.LongTensor(encoder_input_batch_len)
        decoder_input_batch_len = cuda.LongTensor(decoder_input_batch_len)

        encoder_hidden = self.encoder.initHidden(N)
        decoder_hidden = self.decoder.initHidden(N)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        # Encode the sentence
        encoder_output, encoder_hidden = self.encoder(encoder_input_batch, encoder_hidden)

        if self.attention is False:
            last_encoder_states = Variable(cuda.FloatTensor(N, self.encoder.hidden_size).zero_())
            for i in xrange(N):
                last_encoder_states[i] = encoder_output[i, encoder_input_batch_len[i], :]

        else:
            F = encoder_output.contiguous().view(-1, self.encoder.hidden_size) # is NT x E
            self.wf = self.wf_layer(F).view(N, self.max_length, self.D_size) # output is NT x D => N x T x D
            mask = Variable(cuda.FloatTensor(N, self.max_length, 1).zero_(), requires_grad=False)
            for i in xrange(N):
                t = encoder_input_batch_len[i]
                mask[i, :t+1, :] = 1

        del encoder_input_batch
        
        response = []
        loss = 0
        if train is True:
            if self.attention is False:
                decoder_output_batch, decoder_hidden = self.decoder(decoder_input_batch, decoder_hidden, last_encoder_states, question_persona_batch, answer_persona_batch)
            else:
                # decoder_step_input = torch.t(Variable(cuda.LongTensor([[utils.SOS_token]*N]), requires_grad=False))
                decoder_output_batch = []
                for t in xrange(self.max_length):
                    decoder_step_output, decoder_hidden = self.decoder(decoder_input_batch[:, t], decoder_hidden, 
                                                                        encoder_output, self.wf, mask, question_persona_batch, answer_persona_batch)
                    decoder_output_batch.append(decoder_step_output)
                    #input, hidden, encoder_states, wf_mat, p1, p2
                decoder_output_batch = torch.cat(decoder_output_batch, 1)

            for i in xrange(N):
                t = decoder_input_batch_len[i]
                loss += self.criterion(decoder_output_batch[i, :t+1, :], decoder_target_batch[i, :t+1])
        else:
            # greedy decode
            if decode_flag == 'greedy':
                response = [[self.lang.index2word[utils.SOS_token]] for i in xrange(N)]
                decoder_step_input = torch.t(Variable(cuda.LongTensor([[utils.SOS_token]*N]), requires_grad=False)) # To make it N x 1
                for t in xrange(self.max_length):
                    if self.attention is True:
                        decoder_step_output, decoder_hidden = self.decoder(decoder_step_input, decoder_hidden, 
                                                                            encoder_output, self.wf, mask, question_persona_batch, answer_persona_batch)
                    else:
                        decoder_step_output, decoder_hidden = self.decoder(decoder_step_input, decoder_hidden, last_encoder_states, question_persona_batch, answer_persona_batch)
                    decoder_step_output = decoder_step_output.view(N, self.lang.n_words)

                    # Idea is to pick the next word randomly from the probability distribution over the words
                    
                    # Argmax code
                    scores, idx = torch.max(decoder_step_output, 1)
                    decoder_step_input = idx

                    for i in xrange(N):

                        # Argmax code
                        word = self.lang.index2word[idx[i].data[0]]
                        if response[i][-1] != self.lang.index2word[utils.EOS_token]:
                            response[i].append(word)

            # Random sampling code
            elif decode_flag == 'random':
                response = [[self.lang.index2word[utils.SOS_token]] for i in xrange(N)]
                decoder_step_input = torch.t(Variable(cuda.LongTensor([[utils.SOS_token]*N]), requires_grad=False)) # To make it N x 1
                for t in xrange(self.max_length):
                    if self.attention is True:
                        decoder_step_output, decoder_hidden = self.decoder(decoder_step_input, decoder_hidden, 
                                                                            encoder_output, self.wf, mask, question_persona_batch, answer_persona_batch)
                    else:
                        decoder_step_output, decoder_hidden = self.decoder(decoder_step_input, decoder_hidden, last_encoder_states, question_persona_batch, answer_persona_batch)
                    decoder_step_output = decoder_step_output.view(N, self.lang.n_words)

                    # Idea is to pick the next word randomly from the probability distribution over the words

                    # Random sampling code
                    scores, idx = torch.topk(decoder_step_output, random_sample_size, 1)
                    # Convert scores to probabilities from log(p) = p and normalize the top 100 scores
                    p = scores.data.cpu().numpy()
                    p -= np.array([logsumexp(p, 1)]).T # Normalize in a numerically stable way
                    p = np.exp(p) # obtain the probabilities
                    pi = idx.data.cpu().numpy() # obtain the indices in numpy array

                    for i in xrange(N):
                        # Random sampling code
                        ind = np.random.choice(pi[i, :], p=p[i, :]) # Randomly choose based on the probability distribution of scores
                        decoder_step_input[i].data = cuda.LongTensor([ind]) # Make that the next input
                        word = self.lang.index2word[ind]
                        if response[i][-1] != self.lang.index2word[utils.EOS_token]:
                            response[i].append(word)

            # Beam decode
            else:
                assert N == 1 # Beam decode works only on one example at a time

                
                print_gpu_status(382)
                response = [[] for b in xrange(beam_size)]
                # insertion_list = [(-np.float('inf'), utils.SOS_token, decoder_hidden)]
                insertion_list = [(-np.float('inf'), utils.SOS_token, decoder_hidden) for b in xrange(beam_size)]
                # start with -inf 
                for t in xrange(self.max_length):
                    insertion_list_copy = [(-np.float('inf'), utils.SOS_token, decoder_hidden) for b in xrange(beam_size)]
                    # starts with beer and i, and it's probability

                    for (lprob, ind, hstate) in insertion_list:
                        decoder_step_input = torch.t(Variable(cuda.LongTensor([[ind]]), requires_grad=False))
                        if self.attention is True:
                            decoder_step_output, decoder_hidden = self.decoder(decoder_step_input, hstate, encoder_output, self.wf,
                                                                                mask, question_persona_batch, answer_persona_batch)
                        else:
                            decoder_step_output, decoder_hidden = self.decoder(decoder_step_input, hstate, last_encoder_states,
                                                                                question_persona_batch, answer_persona_batch)
                        logprob, idx = torch.topk(decoder_step_output[0][0] + lprob, beam_size) # first sample, first tsteps

                        for (lprob, ind) in zip(logprob, idx):
                            key = lprob
                            insertion_list_copy.append((lprob, ind, decoder_hidden))
                            i = len(insertion_list_copy) - 1
                            while i>0 and insertion_list_copy[i][0] > insertion_list_copy[i-1][0]:
                                insertion_list_copy[i-1] = insertion_list_copy[i]
                                i -= 1
                            if len(insertion_list_copy) > beam_size:
                                del insertion_list_copy[-1]
                            insertion_list_copy = insertion_list_copy[:beam_size]

                    # correctly beer and i added
                    insertion_list = [i for i in insertion_list_copy]
                    del insertion_list_copy 

                

        # Step back
        if train is True:
            # with self.graph.as_default() as graph:
            #     tf.summary.scalar('loss', loss.data[0].cpu().numpy())
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        
        del decoder_target_batch
        del decoder_input_batch
        del encoder_sequence_length
        del decoder_sequence_length
        
        response = [' '.join(resp[1:-1]) for resp in response]
        return response, loss
