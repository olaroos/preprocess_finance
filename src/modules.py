import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module_multiheadattention_tweaked import MultiheadAttentionTweaked


class CopyGateModule(nn.Module):
    def __init__(self, config=None):
        super(CopyGateModule, self).__init__()
        assert config is not None
        # Following Learn what not to question... - Liu et.al. - 2019/02/27
        self.p_value = config["dropout"]
        self.linear1 = nn.Linear(
            config["hidden_size"], config["hidden_size"], bias=True
        )
        self.linear2 = nn.Linear(
            config["hidden_size"], int(config["hidden_size"] / 2.0), bias=True
        )
        self.linear3 = nn.Linear(int(config["hidden_size"] / 2.0), 1, bias=True)

    def original(self, g_t):
        g_t = F.leaky_relu(self.linear1(g_t), negative_slope=0.01)
        g_t = F.leaky_relu(self.linear2(g_t), negative_slope=0.01)
        g_t = F.leaky_relu(self.linear3(g_t), negative_slope=0.01)
        g_t = torch.sigmoid(g_t)
        return g_t

    def forward(self, g_t):
        original_shape = list(g_t.shape)
        original_shape[-1] = 1

        g_t = g_t.view(-1, g_t.shape[-1])
        g_t = F.leaky_relu(
            F.dropout(self.linear1(g_t), p=self.p_value), negative_slope=0.01
        )
        g_t = F.leaky_relu(
            F.dropout(self.linear2(g_t), p=self.p_value), negative_slope=0.01
        )
        g_t = F.leaky_relu(
            F.dropout(self.linear3(g_t), p=self.p_value), negative_slope=0.01
        )
        g_t = torch.sigmoid(g_t)

        return g_t.view(original_shape)


class CustomAttentionLayer(nn.TransformerDecoderLayer):
    def __init__(self, config=None, nhead=None):
        assert config is not None
        nhead = config["number_heads"] if not nhead else nhead

        super(CustomAttentionLayer, self).__init__(
            d_model=config["hidden_size"],
            nhead=nhead,
            dim_feedforward=config["hidden_size"],
            dropout=config["dropout"],
        )

        self.multihead_attn_tweaked = MultiheadAttentionTweaked(
            config["hidden_size"], nhead, dropout=config["dropout"]
        )

    # this layer skip last norm and linear before returning the softmaxed output
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):

        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, _, attn_output_entropy = self.multihead_attn_tweaked(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn_output_entropy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
            adds a positional encoding to input that's intended for use with a Transformer model
            taken from www.pytorch.org
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.div_term = div_term
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.unsqueeze(0)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if x.shape[0] > self.pe.shape[0]:
            print(
                "batch sequence is too long {} to fit positional encoding max-length {}".format(
                    x.shape[0], self.pe.shape[0]
                )
            )
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TagEmbedding(nn.Module):
    def __init__(self, config):
        super(TagEmbedding, self).__init__()
        self.conf = config

        """
        Sizes:

        beginning-inner-outer
        named-entity
        part-of-speech
        answer
        clue-word
        acronyms
        all-capital
        capital-first-letter
        potential-number

        bio:    16
        ner:    16
        pos:    16
        ans:    8
        clue:   8
        acr:    8
        acap:   8
        cap:    8
        pnum:   8
        """

        self.bio_embedding = nn.Embedding(3, 16)
        self.ner_embedding = nn.Embedding(19, 16)
        self.pos_embedding = nn.Embedding(19, 16)
        self.ans_embedding = nn.Embedding(3, 8)
        self.clue_embedding = nn.Embedding(2, 8)

        self.acr_embedding = nn.Embedding(2, 8)
        self.acap_embedding = nn.Embedding(2, 8)
        self.cap_embedding = nn.Embedding(2, 8)
        self.pnum_embedding = nn.Embedding(11, 8)

        self.out_size = (
            self.bio_embedding.weight.shape[-1]
            + self.ner_embedding.weight.shape[-1]
            + self.pos_embedding.weight.shape[-1]
            + self.ans_embedding.weight.shape[-1]
            + self.clue_embedding.weight.shape[-1]
            + self.acr_embedding.weight.shape[-1]
            + self.acap_embedding.weight.shape[-1]
            + self.cap_embedding.weight.shape[-1]
            + self.pnum_embedding.weight.shape[-1]
        )

    def init_weights(self):
        initrange = 0.1
        self.bio_embedding.weight.data.uniform_(-initrange, initrange)
        self.ner_embedding.weight.data.uniform_(-initrange, initrange)
        self.pos_embedding.weight.data.uniform_(-initrange, initrange)
        self.ans_embedding.weight.data.uniform_(-initrange, initrange)
        self.clue_embedding.weight.data.uniform_(-initrange, initrange)
        self.acr_embedding.weight.data.uniform_(-initrange, initrange)
        self.acap_embedding.weight.data.uniform_(-initrange, initrange)
        self.cap_embedding.weight.data.uniform_(-initrange, initrange)
        self.pnum_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch):
        # all tags are set to 0 if they are not created so we don't have to use an if statement
        assert float(torch.max(batch.bio_tag)) < self.bio_embedding.weight.shape[0]
        assert float(torch.max(batch.ner_tag)) < self.ner_embedding.weight.shape[0]
        assert float(torch.max(batch.pos_tag)) < self.pos_embedding.weight.shape[0]
        assert float(torch.max(batch.ans_tag)) < self.ans_embedding.weight.shape[0]
        assert float(torch.max(batch.clue_tag)) < self.clue_embedding.weight.shape[0]
        assert float(torch.max(batch.acr_tag)) < self.acr_embedding.weight.shape[0]
        assert float(torch.max(batch.acap_tag)) < self.acap_embedding.weight.shape[0]
        assert float(torch.max(batch.cap_tag)) < self.cap_embedding.weight.shape[0]
        assert float(torch.max(batch.pnum_tag)) < self.pnum_embedding.weight.shape[0]

        assert float(torch.min(batch.bio_tag)) > -1
        assert float(torch.min(batch.ner_tag)) > -1
        assert float(torch.min(batch.pos_tag)) > -1
        assert float(torch.min(batch.ans_tag)) > -1
        assert float(torch.min(batch.clue_tag)) > -1
        assert float(torch.min(batch.acr_tag)) > -1
        assert float(torch.min(batch.acap_tag)) > -1
        assert float(torch.min(batch.cap_tag)) > -1
        assert float(torch.min(batch.pnum_tag)) > -1

        bio = self.bio_embedding(batch.bio_tag)
        pos = self.pos_embedding(batch.pos_tag)
        ner = self.ner_embedding(batch.ner_tag)
        ans = self.ans_embedding(batch.ans_tag)
        clue = self.clue_embedding(batch.clue_tag)
        acr = self.acr_embedding(batch.acr_tag)
        acap = self.acap_embedding(batch.acap_tag)
        cap = self.cap_embedding(batch.cap_tag)
        pnum = self.pnum_embedding(batch.pnum_tag)

        output = torch.cat((bio, pos, ner, ans, clue, acr, acap, cap, pnum), -1)

        return output


class LossSoftmax(nn.Module):
    def __init__(self):
        super(LossSoftmax, self).__init__()

    def no_loss(self, entropy):
        assert entropy.dim() == 2
        logprob = F.log_softmax(entropy, dim=1)
        return logprob

    def loss(self, entropy, target=None, K=1):
        logprob = self.no_loss(entropy)
        loss = F.nll_loss(logprob, target, ignore_index=0, reduction="none")
        logprobk, widxk = torch.topk(logprob, K)
        accu = self.accu(loss, widxk, target=target)
        return loss, accu, widxk, entropy, logprob

    def accu(self, loss, widx, target=None):
        assert widx.dim() == 2
        assert widx.shape[1] == 1
        accu = torch.eq(widx.squeeze(1).cpu(), target.cpu()).float()
        mask = ~torch.eq(target.cpu().float(), torch.zeros(target.shape))
        mask = mask.float()
        return accu * mask
