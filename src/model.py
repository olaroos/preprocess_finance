import torch.nn.functional as F
import torch.nn as nn
import torch
import random
from src.modules import (
    PositionalEncoding,
    LossSoftmax,
)
from src.fun import (get_data,
                     split_data)
config = {'bs': 8,
          'n_layers': 6,
          'd_model': 10,
          'nhead':16,
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

        self.post_encoder = nn.Linear(d_model * seq_len, 2)
        self.softmax = LossSoftmax()

        # Seed
        self.seed(seed=config["seed"])
        self.init_weights()

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

    def forward(self, batch, config=None):
        assert config is not None

        src, tgt = batch[0], batch[1]

        """pre-encoder block"""

        src = self.sequence_to_first_dimension(src, batch=batch)
        src = self.encoder_pos_encoder(src)

        """encoder block"""
        # no self-attention mask needed only padding between sequences in batch
        memory = self.transformer_encoder(src, mask=None)
        memory = memory.view(memory.shape[0] * memory.shape[1], -1)
        output = self.post_encoder(memory)


        """calculate loss"""
        logprob = self.eps_log_softmax(output, dim=2)
        target = self.sequence_to_first_dimension(
            tgt, batch=batch
        )
        loss = F.nll_loss(input=logprob.view(-1, logprob.shape[-1]), target=target.view(-1))

        """statistics"""
        loss_stats = self.get_loss_stats(
            batch=batch,
            loss=loss,
            logprob=logprob,
            target=target,

        )
        return loss_stats


    def get_loss_stats(
        self,
        batch=None,
        loss=None,
        logprob=None,
        target=None,
        copy_gate=None,
        decoder_dict=None,
    ):
        assert decoder_dict is not None
        accu = self.get_logprob_accu(
            logprob=logprob, target=target, decoder_dict=decoder_dict
        )
        denominator = torch.pow(batch.tgt_len.float(), -1)
        accu = float(torch.sum(accu.sum(0).squeeze() * denominator) / batch.bs)

        copy_accu = self.get_copy_accu(logprob, batch)
        gate_accu = self.get_gate_accu(copy_gate, batch)
        gate_level = self.get_gate_level(copy_gate, batch)

        loss_dict = {
            "loss": loss,
            "accu": accu,
            "copy_accu": copy_accu,
            "gate_accu": gate_accu,
            "gate_level": gate_level,
        }

        return loss_dict

    def get_pre_decoder_block(
        self,
        idx_emb=None,
        prev_word_copied=None,
        tgt_pos_tag=None,
        tgt_copy_pos=None,
        tgt_emb=None,
    ):
        return torch.cat(
            (
                self.calc_idx_emb(idx_emb),
                self.calc_previous_step_copy(prev_word_copied),
                self.calc_tgt_pos_tag(tgt_pos_tag),
                self.calc_pos_copy_emb(tgt_copy_pos),
                tgt_emb,
            ),
            -1,
        )

    def get_start_of_decoder_block(self, batch=None):
        assert batch is not None
        return self.get_pre_decoder_block(
            idx_emb=batch.tgt_input_idx[:, :1],
            prev_word_copied=batch.tgt_previous_word_copied[:, :1],
            tgt_pos_tag=batch.tgt_pos_tag[:, :1],
            tgt_copy_pos=batch.tgt_copy_pos[:, :1],
            tgt_emb=batch.tgt_emb[:, :1],
        )

    def generate_text(
        self,
        batch,
        config=None,
        encoder_dict=None,
        decoder_dict=None,
        loader=None,
        to_cuda=True,
    ):
        assert batch.bs == 1
        assert config is not None
        assert encoder_dict is not None
        assert decoder_dict is not None
        src_key_mask, tgt_key_mask = self.make_key_masks(batch, to_cuda=to_cuda)
        src = self.get_pre_encoder_block(batch=batch, encoder_dict=encoder_dict)
        src = self.sequence_to_first_dimension(src, batch=batch)
        src = self.encoder_pos_encoder(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_mask)
        tmp_tgt_input = self.get_start_of_decoder_block(batch=batch)
        persistent_tgt_input = tmp_tgt_input.cpu()
        text = []
        for i in range(1, int(batch.tgt_len)):
            topk = self.generate_topk(
                tmp_tgt_input=tmp_tgt_input,
                batch=batch,
                memory=memory,
                src_key_mask=src_key_mask,
                tgt_key_mask=tgt_key_mask,
                to_cuda=to_cuda,
            )
            topk = topk[-1, :, 0]
            (
                idx_emb,
                copied,
                tgt_pos_tag,
                tgt_copy_pos_tag,
                tgt_emb,
                word_list,
                copy_word_list,
            ) = idx_to_tensor(
                topk.cpu(), batch=batch, decoder_dict=decoder_dict, loader=loader
            )
            generated_tmp_tgt_input = self.get_pre_decoder_block(
                idx_emb=idx_emb if not to_cuda else idx_emb.cuda(),
                prev_word_copied=copied.long() if not to_cuda else copied.long().cuda(),
                tgt_pos_tag=tgt_pos_tag if not to_cuda else tgt_pos_tag.cuda(),
                tgt_copy_pos=tgt_copy_pos_tag
                if not to_cuda
                else tgt_copy_pos_tag.cuda(),
                tgt_emb=tgt_emb if not to_cuda else tgt_emb.cuda(),
            )
            persistent_tgt_input = torch.cat(
                (persistent_tgt_input, generated_tmp_tgt_input.cpu()), dim=1
            )
            # put input for next step on the GPU
            tmp_tgt_input = (
                persistent_tgt_input if not to_cuda else persistent_tgt_input.cuda()
            )
            word = word_list[0] if word_list[0] is not None else copy_word_list[0]
            text.append(word)
        return text

    def generate_topk(
        self,
        tmp_tgt_input=None,
        batch=None,
        memory=None,
        src_key_mask=None,
        tgt_key_mask=None,
        to_cuda=True,
    ):

        generated_len = tmp_tgt_input.shape[1]
        tgt_input = self.sequence_to_first_dimension(tmp_tgt_input, batch=batch)
        tgt_input = self.decoder_pos_encoder(tgt_input)
        output_0 = self.get_decoder_0(
            input=tgt_input,
            memory=memory,
            src_key_mask=src_key_mask,
            tgt_key_mask=tgt_key_mask[:, :generated_len],
            to_cuda=to_cuda,
        )

        prob_generate = self.post_decoder(output_0)
        prob_copy = self.get_prob_copy(
            batch=batch,
            input=output_0,
            memory=memory,
            src_key_mask=src_key_mask,
            tgt_key_mask=tgt_key_mask[:, :generated_len],
            to_cuda=to_cuda,
        )
        copy_gate = self.copy_gate(output_0)

        # merge probabilities
        assert prob_generate.shape[0:1] == copy_gate.shape[0:1]
        assert prob_copy.shape[0:1] == copy_gate.shape[0:1]
        prob_generate = prob_generate * (1 - copy_gate)
        prob_copy = prob_copy * copy_gate
        output = torch.cat((prob_generate, prob_copy), -1)
        # probabilities of decoder indexes
        logprob = self.eps_log_softmax(output, dim=2)
        _, topk = torch.topk(logprob, 1)
        return topk

    def get_generate_slice_map(self, generate_map: list):
        generate_slice_map = []
        start = 0
        end = 1
        for i, generate in enumerate(generate_map):
            if generate:
                if start < i:
                    generate_slice_map.append(slice(start, end))
                generate_slice_map.append(i)
                start = i + 1
                end = start + 1
            else:
                end = i + 1
        if start != len(generate_map):
            generate_slice_map.append(slice(start, end))
        return generate_slice_map

    def generate_decoder_input(
        self,
        batch,
        memory=None,
        decoder_dict=None,
        loader=None,
        src_key_mask=None,
        tgt_key_mask=None,
    ):
        self.training = False
        teacher_forcing_cutoff = self.tf_ratio

        # create map for generation
        generate_map = [
            1 if (random.random() > teacher_forcing_cutoff) else 0
            for _ in range(int(max(batch.tgt_len)))
        ]
        slice_map = self.get_generate_slice_map(generate_map)

        # TODO: verify somehow that the generated parts and the parts taken from batch are concatenated in a correct way
        with torch.no_grad():
            s = slice_map[0] if (type(slice_map[0]) == slice) else slice(0, 1)
            # setting the persistent input to be on the cpu to clear the tmp to save GPU-RAM
            accumulated_idx_emb = batch.tgt_input_idx[:, s].cpu()
            accumulated_copied = batch.tgt_previous_word_copied[:, s].long().cpu()
            accumulated_tgt_pos_tag = batch.tgt_pos_tag[:, s].cpu()
            accumulated_tgt_copy_pos = batch.tgt_copy_pos[:, s].cpu()
            accumulated_tgt_emb = batch.tgt_emb[:, s].cpu()
            tmp_tgt_input = self.get_pre_decoder_block(
                idx_emb=accumulated_idx_emb.cuda(),
                prev_word_copied=accumulated_copied.long().cuda(),
                tgt_pos_tag=accumulated_tgt_pos_tag.cuda(),
                tgt_copy_pos=accumulated_tgt_copy_pos.cuda(),
                tgt_emb=accumulated_tgt_emb.cuda(),
            )
            accumulated_tgt_input = tmp_tgt_input.cpu()
            for s in slice_map[1:]:
                if type(s) == int:
                    # if we encounter an index we generate the next input
                    topk = self.generate_topk(
                        tmp_tgt_input=tmp_tgt_input,
                        batch=batch,
                        memory=memory,
                        src_key_mask=src_key_mask,
                        tgt_key_mask=tgt_key_mask,
                    )
                    # clear memory, not certain if this actually helps
                    del tmp_tgt_input
                    torch.cuda.empty_cache()

                    # use only last generated indexes
                    topk = topk[-1, :, 0]
                    (
                        idx_emb,
                        copied,
                        tgt_pos_tag,
                        tgt_copy_pos_tag,
                        tgt_emb,
                        word_list,
                        copy_word_list,
                    ) = idx_to_tensor(
                        topk.cpu(),
                        batch=batch,
                        decoder_dict=decoder_dict,
                        loader=loader,
                    )
                    accumulated_idx_emb = torch.cat(
                        (accumulated_idx_emb, idx_emb.cpu()), 1
                    )
                    accumulated_copied = torch.cat(
                        (accumulated_copied, copied.long().cpu()), 1
                    )
                    accumulated_tgt_pos_tag = torch.cat(
                        (accumulated_tgt_pos_tag, tgt_pos_tag.cpu()), 1
                    )
                    accumulated_tgt_copy_pos = torch.cat(
                        (accumulated_tgt_copy_pos, tgt_copy_pos_tag.cpu()), 1
                    )
                    accumulated_tgt_emb = torch.cat(
                        (accumulated_tgt_emb, tgt_emb.cpu()), 1
                    )

                    generated_tmp_tgt_input = self.get_pre_decoder_block(
                        idx_emb=idx_emb.cuda(),
                        prev_word_copied=copied.long().cuda(),
                        tgt_pos_tag=tgt_pos_tag.cuda(),
                        tgt_copy_pos=tgt_copy_pos_tag.cuda(),
                        tgt_emb=tgt_emb.cuda(),
                    )
                else:
                    # TODO: assert that this is the correct index we get from batch when running teacher-forcing
                    idx_emb = batch.tgt_input_idx[:, s]
                    copied = batch.tgt_previous_word_copied[:, s]
                    tgt_pos_tag = batch.tgt_pos_tag[:, s]
                    tgt_copy_pos = batch.tgt_copy_pos[:, s]
                    tgt_emb = batch.tgt_emb[:, s]
                    generated_tmp_tgt_input = self.get_pre_decoder_block(
                        idx_emb=idx_emb,
                        prev_word_copied=copied,
                        tgt_pos_tag=tgt_pos_tag,
                        tgt_copy_pos=tgt_copy_pos,
                        tgt_emb=tgt_emb,
                    )
                    accumulated_idx_emb = torch.cat(
                        (accumulated_idx_emb, idx_emb.cpu()), 1
                    )
                    accumulated_copied = torch.cat(
                        (accumulated_copied, copied.long().cpu()), 1
                    )
                    accumulated_tgt_pos_tag = torch.cat(
                        (accumulated_tgt_pos_tag, tgt_pos_tag.cpu()), 1
                    )
                    accumulated_tgt_copy_pos = torch.cat(
                        (accumulated_tgt_copy_pos, tgt_copy_pos.cpu()), 1
                    )
                    accumulated_tgt_emb = torch.cat(
                        (accumulated_tgt_emb, tgt_emb.cpu()), 1
                    )

                # add new input to previous input
                accumulated_tgt_input = torch.cat(
                    (accumulated_tgt_input, generated_tmp_tgt_input.cpu()), dim=1
                )

                # put input for next step on the GPU
                tmp_tgt_input = accumulated_tgt_input.cuda()

        self.training = True
        generated_tgt_input = self.get_pre_decoder_block(
            idx_emb=accumulated_idx_emb.cuda(),
            prev_word_copied=accumulated_copied.cuda(),
            tgt_pos_tag=accumulated_tgt_pos_tag.cuda(),
            tgt_copy_pos=accumulated_tgt_copy_pos.cuda(),
            tgt_emb=accumulated_tgt_emb.cuda(),
        )
        return generated_tgt_input


    def validation_loss(
        self, config=None, loader=None, encoder_dict=None, decoder_dict=None
    ):
        assert config is not None
        assert loader is not None
        assert encoder_dict is not None
        assert decoder_dict is not None
        self.eval()
        with torch.no_grad():
            vloss = 0
            vaccu = 0
            vcopy_accu = 0
            vgate_level = 0
            generator = loader.batch_generator(
                bs=config["valid_bs"],
                _batch_list=loader.valid_batchtokens,
                encoder_dict=encoder_dict,
                decoder_dict=decoder_dict,
            )
            for i, batch in enumerate(generator):
                vloss_dict = self.forward(
                    batch=batch,
                    config=config,
                    encoder_dict=encoder_dict,
                    decoder_dict=decoder_dict,
                    loader=loader,
                )
                vloss += vloss_dict["loss"]
                vaccu += vloss_dict["accu"]
                vcopy_accu += vloss_dict["copy_accu"]
                vgate_level += vloss_dict["gate_level"]
            vloss /= i + 1
            vaccu /= i + 1
            vcopy_accu /= i + 1
            vgate_level /= i + 1

        del generator
        self.train()
        return {
            "vloss": vloss,
            "vaccu": vaccu,
            "vcopy_accu": vcopy_accu,
            "vgate_level": vgate_level,
        }

    def get_pre_decoder_block_generation(self, batch, tgt_input_raw):
        raise NotImplementedError
        tgt_input = self.decoder_idx_embedding(tgt_input_raw)
        tgt_input_embedding = torch.zeros(
            tgt_input_raw.size(0), 1, batch.tgt_emb.shape[-1]
        )

        for t, idx in enumerate(tgt_input_raw):
            if idx == self.decoder_dict.pad_token_idx:
                tgt_input[idx:] = torch.zeros(
                    tgt_input_raw.shape[0], 1, tgt_input.shape[-1]
                )
                break
            tgt_input_embedding[t, :] = torch.tensor(
                self.decoder_dict.idx_to_emb(int(idx))
            )

        tgt_input = torch.cat((tgt_input, tgt_input_embedding.cuda()), -1)
        return tgt_input

    def get_gate_level(self, g_t, batch):
        """
           measure the average gate value that is guessed for words that should be generated
        """
        gate = g_t.detach()
        if gate.dim() > 2:
            gate = gate.squeeze(-1)

        gate_level = gate * batch.tgt_copy_gate.transpose(0, 1).contiguous()
        gate_denominator = float(torch.sum(batch.tgt_copy_gate)) ** -1
        gate_level = float(torch.sum(gate_level)) * gate_denominator

        return gate_level

    def get_copy_accu(self, logprob, batch):

        target = batch.tgt_copy_output_idx.transpose(0, 1).contiguous()
        target_gate = batch.tgt_copy_gate.transpose(0, 1)

        target = torch.where(
            target_gate == 1.0, target, torch.zeros(target.shape).long().cuda()
        )

        _, widx_topk = torch.topk(logprob, 1)
        copy_accu = torch.eq(widx_topk.squeeze(-1).cpu(), target.cpu()).float()

        mask_out_generated = (
            torch.eq(target_gate.cpu().float(), torch.ones(target_gate.shape))
        ).float()
        copy_accu = copy_accu * mask_out_generated

        copy_denominator = float(torch.sum(batch.tgt_copy_gate)) ** -1
        return float(torch.sum(copy_accu)) * copy_denominator

    def get_gate_accu(self, g_t, batch, cutoff=0.5):
        gate = g_t.detach()
        if gate.dim() > 2:
            gate = gate.squeeze(-1)

        gate_result = torch.where(
            gate > cutoff,
            torch.ones(gate.shape).float().cuda(),
            torch.zeros(gate.shape).float().cuda(),
        )

        target = batch.tgt_copy_gate.transpose(0, 1).contiguous()

        gate_accu = torch.eq(gate_result, target).float()
        # remove guesses of zeros from gate_accu
        mask_generate_positions = (
            torch.eq(target.cpu().float(), torch.ones(target.shape))
        ).float()

        gate_accu = gate_accu * mask_generate_positions.cuda()

        gate_denominator = float(torch.sum(batch.tgt_copy_gate)) ** -1
        gate_accu = float(torch.sum(gate_accu)) * gate_denominator
        return gate_accu

    def get_logprob_accu(self, logprob=None, target=None, decoder_dict=None):
        assert logprob is not None
        assert target is not None
        assert decoder_dict is not None
        _, widx_topk = torch.topk(logprob, 1)
        accu = torch.eq(widx_topk.squeeze(-1).cpu(), target.cpu()).float()
        # mask out the padded elements
        pad_mask = (
            ~torch.eq(
                target.cpu().float(),
                torch.ones(target.shape) * decoder_dict.pad_token_idx,
            )
        ).float()
        assert accu.shape == pad_mask.shape
        return accu * pad_mask

    def eps_log_softmax(self, entropy=None, dim=2):
        assert entropy is not None
        assert entropy.dim() == 3
        eps = 10 ** -6
        entropy = F.softmax(entropy, dim=dim)
        logprob = torch.log(entropy + eps)
        return logprob

loader = batch_generator(data=training_data)