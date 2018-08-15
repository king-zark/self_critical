from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .CaptionModel import CaptionModel
from .AttModel import pack_wrapper


class MergeAttentionModel(CaptionModel):
    def __init__(self, opt):
        super(MergeAttentionModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.merge_size = 512
        self.is_hard_attention = True

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.fc_embed = nn.Linear(self.fc_feat_size, self.merge_size)
        self.att_embed = nn.Linear(self.att_feat_size, self.merge_size)
        self.rnn_embed = nn.Linear(self.rnn_size, self.merge_size)
        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.merge_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.merge_size, self.merge_size), nn.ReLU(), nn.Dropout(0.5)] for _ in
                          range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(
                *(reduce(lambda x, y: x + y, self.logit) + [nn.Linear(self.merge_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size,
                                                      self.rnn_size, self.num_layers, bias=False,
                                                      dropout=self.drop_prob_lm)

    def core(self, xt, fc_embeds, att_embeds, p_att_feats, att_masks, state):
        if self.is_hard_attention:
            return self.hard_core(xt, fc_embeds, att_embeds, p_att_feats, att_masks, state)
        else:
            return self.soft_core(xt, fc_embeds, att_embeds, p_att_feats, att_masks, state)

    def soft_core(self, xt, fc_embeds, att_embeds, p_att_feats, att_masks, state):
        rnn_feats, state = self.rnn(xt.unsqueeze(0), state)
        rnn_embeds = self.rnn_embed(rnn_feats.squeeze(0))
        weights = self.attend(p_att_feats, att_masks, state)  # [batch, att_size]
        att_res = torch.bmm(weights.unsqueeze(1), att_embeds).squeeze(1)  # batch * att_feat_size
        log_prob = F.log_softmax(self.logit(F.relu(rnn_embeds + fc_embeds + att_res)))
        return log_prob, state

    def hard_core(self, xt, fc_embeds, att_embeds, p_att_feats, att_masks, state):
        rnn_feats, state = self.rnn(xt.unsqueeze(0), state)
        rnn_embeds = self.rnn_embed(rnn_feats).squeeze(0)
        weights = self.attend(p_att_feats, att_masks, state)  # [batch, att_size]
        predicts = self.predict(fc_embeds, att_embeds, rnn_embeds)  # [batch_size, att_size, vocab_size + 1]
        predicts = torch.bmm(weights.unsqueeze(1), predicts).squeeze(1)
        log_prob = torch.log(predicts + 1e-10)
        return log_prob, state

    def attend(self, p_att_feats, att_masks, state):
        att_size = p_att_feats.size(1)
        att_h = self.h2att(state[0][-1])  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(p_att_feats)  # batch * att_size * att_hid_size
        dot = p_att_feats + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = self.alpha_net(dot)  # batch * att_size * 1
        dot = dot.view(-1, att_size)  # batch * att_size
        weights = F.softmax(dot)
        if att_masks is not None:
            weights = weights * att_masks.view(-1, att_size).float()
            weights = weights / weights.sum(1, keepdim=True)  # normalize to 1
        return weights

    def predict(self, fc_embeds, att_embeds, rnn_embeds):
        other_embeds = fc_embeds + rnn_embeds
        other_embeds = other_embeds.unsqueeze(1).expand_as(att_embeds)
        all_embeds = att_embeds + other_embeds
        logits = self.logit(F.relu(all_embeds))
        predicts = F.softmax(logits, dim=-1)
        return predicts

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        p_fc_feats = self.fc_embed(fc_feats)
        p_att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        pp_att_feats = pack_wrapper(self.ctx2att, att_feats, att_masks)

        return p_fc_feats, p_att_feats, pp_att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)
        logprobs, state = self.core(xt, fc_feats, att_feats, p_att_feats, att_masks, state)
        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k + 1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k + 1].expand(*((beam_size,) + pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k + 1].expand(
                *((beam_size,) + p_att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                          tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                  tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs
