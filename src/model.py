import torch.nn.functional as F
import torch.nn as nn
import torch
import random
from src.modules import (
    PositionalEncoding,
    LossSoftmax,
)
from src.fun import (get_data,
                     split_data,
                     batch_generator)
config = {'seed': 42,
          'dropout': 0.1,
          'bs': 8,
          'n_layers': 6,
          'd_model': 22,
          'nhead':22,
          'sequence_length': 10,
          'tgt_key': 'rule_closeup',
          'forbidden_keys': ['date',
                             'up']}

class TransformerModel(nn.Module):
    def __init__(self, config=None, encoder_dict=None, decoder_dict=None):
        assert config is not None
        super(TransformerModel, self).__init__()
        # Setting bs here, used to verify that dimensions are
        # correct when switching sequence and batch-size in tensors
        self.bs = config["bs"]

        # Dimensions
        d_model = config["d_model"]
        nhead = config["nhead"]
        nlayers = config["n_layers"]
        dropout = config["dropout"]
        seq_len = config["sequence_length"]

        self.positional_encoder = PositionalEncoding(d_model=d_model, max_len=128)


        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=2048,
                                                   dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        self.post_encoder = nn.Linear(d_model * seq_len, 1)
        self.softmax = LossSoftmax()

        # Seed
        self.seed(seed=config["seed"])
        self.cuda()
    def seed(self, seed=42):
        """
            Note: not verified if all these commands need to be run for proper seeding
        """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def sequence_to_first_dimension(self, tensor, bs=None):
        assert tensor.shape[0] == bs
        return tensor.transpose(0, 1).contiguous()

    def bs_to_first_dimension(self, tensor, bs=None):
        assert tensor.shape[1] == bs
        return tensor.transpose(0, 1).contiguous()

    def forward(self, src=None, tgt=None, config=None):
        assert config is not None


        """pre-encoder block"""
        src = self.sequence_to_first_dimension(src, bs=self.bs)
        src = self.positional_encoder(src)

        """encoder block"""
        # no self-attention mask needed only padding between sequences in batch
        memory = self.transformer_encoder(src, mask=None)
        memory = self.bs_to_first_dimension(memory, bs=self.bs)
        memory = memory.view(-1, memory.shape[1] * memory.shape[2])

        output = self.post_encoder(memory)
        """calculate loss"""
        logprob = self.eps_log_softmax(output, dim=2)
        target = self.sequence_to_first_dimension(tgt, bs=self.bs)
        loss = F.nll_loss(input=logprob.view(-1, logprob.shape[-1]), target=target.view(-1))

        return loss


data = get_data(config)
training_data, valid_data = split_data(data)
loader = batch_generator(data=training_data)
src, tgt = next(loader)
# print(src.shape)
# print(tgt.shape)

model = TransformerModel(config=config)

model(src=src, tgt=tgt, config=config)
