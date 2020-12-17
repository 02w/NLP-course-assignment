import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_ids = torch.argmax(vec, dim=1).view(-1, 1)
    max_score = torch.gather(vec, 1, max_ids)
    max_score_broadcast = max_score.view(vec.size()[0], -1).repeat(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1).unsqueeze(1))


class Model(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout):
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=1, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix['<START>'], :] = -10000
        self.transitions.data[:, tag_to_ix['<END>']] = -10000

    def _forward_alg(self, feats, lengths):
        if len(feats.size()) == 2:
            feats = feats.unsqueeze(0)
        # Initialize the viterbi variables in log space

        batch_size = feats.size()[0]

        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((batch_size, self.tagset_size), -10000.)
        if torch.cuda.is_available():
            init_alphas = init_alphas.cuda()
        # '<START>' has all of the score.

        init_alphas[:, self.tag_to_ix['<START>']] = 0
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        forward_var_table = torch.zeros((len(lengths), max(lengths), self.tagset_size))
        if torch.cuda.is_available():
            forward_var_table = forward_var_table.cuda()

        # Iterate through the sentence
        for i in range(max(lengths)):
            feat = feats[:, i, :]
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                temp = feat[:, next_tag]
                temp = temp.view(batch_size, -1)
                temp = temp.repeat(1, self.tagset_size)
                emit_score = temp
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag]

                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                result = log_sum_exp(next_tag_var)

                alphas_t.append(result)
            forward_var = torch.cat(alphas_t, 1)
            forward_var_table[:, i, :] = forward_var

        index = torch.tensor(lengths).reshape(len(lengths), 1, 1).repeat(1, 1, self.tagset_size) - 1
        if torch.cuda.is_available():
            index = index.cuda()

        final_forward_var = torch.gather(forward_var_table, dim=1, index=index)
        terminal_var = final_forward_var + self.transitions[self.tag_to_ix['<END>']]
        terminal_var = terminal_var.squeeze()
        alpha = log_sum_exp(terminal_var).squeeze(1)
        return alpha

    def _get_lstm_features(self, sentence, lengths):
        if len(sentence) > 1:
            embeds = self.word_embeds(sentence)
            # lstm_out, _ = self.lstm(embeds)
            embeds = self.drop(embeds)
            embeds = pack_padded_sequence(embeds, lengths=lengths, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(embeds)
            lstm_out = pad_packed_sequence(lstm_out, batch_first=True)[0]
            lstm_out = self.drop(lstm_out)
            lstm_feats = self.hidden2tag(lstm_out)
        else:
            embeds = self.word_embeds(sentence).view(sentence.size()[1], 1, -1)
            lstm_out, _ = self.lstm(embeds)
            lstm_out = lstm_out.view(sentence.size()[1], self.hidden_dim)
            lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags, lengths):
        # Gives the score of a provided tag sequence
        cre, _ = self.mask_matrix(lengths)
        score = torch.zeros((feats.size()[0], 1))
        if torch.cuda.is_available():
            score = score.cuda()
        start = torch.ones((feats.size()[0], 1), dtype=torch.long) * self.tag_to_ix['<START>']
        if torch.cuda.is_available():
            start = start.cuda()
        tags = torch.cat([start, tags], dim=1)
        for i in range(max(lengths)):
            feat = feats[:, i, :]
            trans_ = self.transitions[tags[:, i + 1]]
            trans_ = torch.gather(trans_, 1, tags[:, i].unsqueeze(1))
            trans = cre[:, i].unsqueeze(1) * trans_
            idxs = tags[:, i + 1].unsqueeze(1)
            emit_ = torch.gather(feat, 1, idxs)
            emit = cre[:, i].unsqueeze(1) * emit_
            score = score + trans + emit

        if torch.cuda.is_available():
            last_tags = torch.gather(tags, 1, torch.tensor(lengths).cuda().unsqueeze(1))
        else:
            last_tags = torch.gather(tags, 1, torch.tensor(lengths).unsqueeze(1))
        last_score = self.transitions[self.tag_to_ix['<END>'], last_tags]
        score = score + last_score
        return score.squeeze(1)

    def mask_matrix(self, lengths):
        cre = torch.zeros((len(lengths), max(lengths)))
        cre_matrix = torch.ones((len(lengths), max(lengths), len(self.tag_to_ix)))
        if torch.cuda.is_available():
            cre = cre.cuda()
            cre_matrix = cre_matrix.cuda()
        for i, lens in enumerate(lengths):
            one = torch.zeros(len(self.tag_to_ix))
            one[-1] = 1
            cre[i][:lens] = 1
            cre_matrix[i][lens - 1:] = one

        return cre, cre_matrix

    def _viterbi_decode(self, feats, lengths):
        if len(feats.size()) == 2:
            feats = feats.unsqueeze(0)

        # Initialize the viterbi variables in log space
        batch_size = feats.size()[0]
        init_vvars = torch.full((batch_size, self.tagset_size), -10000.)
        if torch.cuda.is_available():
            init_vvars = init_vvars.cuda()
        init_vvars[:, self.tag_to_ix['<START>']] = 0
        backpointers = []

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars  # (B, T)

        forward_var_table = torch.zeros((len(lengths), max(lengths), self.tagset_size))  # (B, L, T)
        if torch.cuda.is_available():
            forward_var_table = forward_var_table.cuda()
        for i in range(max(lengths)):
            feat = feats[:, i, :]
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)

                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var, dim=1).unsqueeze(1)
                bptrs_t.append(best_tag_id)
                best_node_score = torch.gather(next_tag_var, dim=1, index=best_tag_id)

                viterbivars_t.append(best_node_score)

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = torch.cat(viterbivars_t, 1) + feat
            forward_var_table[:, i, :] = forward_var
            backpointers.append(torch.cat(bptrs_t, 1))

        index = torch.tensor(lengths).reshape(len(lengths), 1, 1).repeat(1, 1, self.tagset_size) - 1
        if torch.cuda.is_available():
            index = index.cuda()

        final_forward_var = torch.gather(forward_var_table, dim=1, index=index)

        # Transition to '<END>'
        terminal_var = final_forward_var + self.transitions[self.tag_to_ix['<END>']]
        terminal_var = terminal_var.squeeze(1)
        best_tag_id = torch.argmax(terminal_var, dim=1).unsqueeze(1)
        path_score = torch.gather(terminal_var, dim=1, index=best_tag_id)
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]

        backpointers_mat = torch.cat(backpointers, 1).reshape(feats.size()[0], -1, self.tagset_size)  #
        backpointers_pad = torch.ones(len(lengths), max(lengths), 1).long() * (self.tagset_size)
        if torch.cuda.is_available():
            backpointers_pad = backpointers_pad.cuda()
        backpointers_mat = torch.cat([backpointers_mat, backpointers_pad], -1)

        for i, length in enumerate(lengths):
            backpointers_mat[i][max(lengths) - length:] = backpointers_mat.clone()[i][:length]
            backpointers_mat[i][:max(lengths) - length] = self.tagset_size
        for i in range(backpointers_mat.size()[1] - 1, -1, -1):
            bptrs_t = backpointers_mat[:, i, :]
            best_tag_id = torch.gather(bptrs_t, dim=1, index=best_tag_id)
            best_path.append(best_tag_id)

        best_path.reverse()
        best_path = torch.cat(best_path, 1)
        last_paths = []

        for i in range(len(lengths)):
            path = best_path[i][max(lengths) - lengths[i]:]
            if path[0] == self.tag_to_ix['<START>']:
                last_paths.append(path[1:])
            else:
                last_paths.append(None)

        return path_score, last_paths

    def neg_log_likelihood(self, sentence, tags, lengths):
        feats = self._get_lstm_features(sentence, lengths)
        forward_score = self._forward_alg(feats, lengths)
        gold_score = self._score_sentence(feats, tags, lengths)
        return torch.mean(forward_score - gold_score)

    def forward(self, sentence, lengths):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, lengths)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats, lengths)
        return score, tag_seq
