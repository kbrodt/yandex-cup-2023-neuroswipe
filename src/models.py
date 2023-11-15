import math

import torch
import torch.nn as nn


class Board(nn.Module):
    def __init__(self, m=4, n=12, emb_size=128, pad_i=0.0, weight=None):
        super().__init__()

        if weight is None:
            weight = torch.empty(
                (1, emb_size, m, n),
                requires_grad=True,
            )
            self.weight = nn.Parameter(weight)
            self.reset_parameters()
        else:
            self.weight = nn.Parameter(weight)

        self.pad_i = pad_i

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, inputs):
        xy, lens = inputs
        # xy: [b, 2]

        xy = xy.unsqueeze(0).unsqueeze(0)  # [1, 1, b, 2]
        out = torch.nn.functional.grid_sample(
            input=self.weight,  # [1, c, m, n]
            grid=xy,  # [1, 1, b, 2]
            align_corners=False,#True,
        )  # [1, c, 1, b]
        out = out.squeeze((0, 2))  # [c, b]
        out = out.transpose(0, 1)  # [b, c]
        out = out.split(lens, dim=0)  # [b * [t, c]]
        t = max(map(len, out))
        out = torch.stack(
            [
                torch.nn.functional.pad(
                    o,
                    (0, 0, 0, t - o.shape[0]),
                    mode="constant",
                    value=self.pad_i,
                )
                for o in out
            ],
            dim=0,
        ) # [b, t, c]

        return out


class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, padding_idx=0, num_layers=1, emb=None, dropout_p=0.1):
        super().__init__()

        self.hidden_size = hidden_size

        if emb is None:
            self.embedding = nn.Embedding(
                input_size,
                emb_size,
                padding_idx=padding_idx,
            )
        else:
            self.embedding = emb

        self.gru = nn.LSTM(
            emb_size,
            hidden_size // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)

        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys, mask):
        # [b, 1, d] -> [b, 1, D] -> [b, t, 1] -> [b]
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        scores[mask.unsqueeze(1)] = -torch.inf

        weights = scores.softmax(dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, emb_size, output_size, max_len, sos_idx, padding_idx=0, emb=None, num_layers=1, dropout_p=0.1):
        super().__init__()

        if emb is None:
            self.embedding = nn.Embedding(
                output_size,
                emb_size,
                padding_idx=padding_idx,
            )
        else:
            self.embedding = emb

        self.attention = BahdanauAttention(hidden_size)
        self.num_layers = num_layers
        self.gru = nn.LSTM(
            emb_size + hidden_size,
            hidden_size // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.max_len = max_len
        self.sos_idx = sos_idx

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, mask=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = encoder_outputs.new_empty(batch_size, 1, dtype=torch.long).fill_(self.sos_idx)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        max_len = self.max_len if target_tensor is None else target_tensor.size(1)

        for i in range(max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, mask=mask,
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs, mask=None):
        embedded =  self.dropout(self.embedding(input))

        if isinstance(hidden, tuple):
            hidden, c = hidden
            # [l * n, b, d] -> [b, d * n, c]
            b = hidden.size(1)
            d = hidden.size(2)
            if self.num_layers > 1:
                query = hidden.permute(1, 0, 2).view(b, self.num_layers, -1, d)[:, -1]
            else:
                query = hidden.permute(1, 0, 2)
            query = query.flatten(1).unsqueeze(1)
            hidden = hidden, c
        else:
            b = hidden.size(1)
            d = hidden.size(2)
            if self.num_layers > 1:
                query = hidden.permute(1, 0, 2).view(b, self.num_layers, -1, d)[:, -1]
            else:
                query = hidden.permute(1, 0, 2)
            query = query.flatten(1).unsqueeze(1)

        context, attn_weights = self.attention(query, encoder_outputs, mask=mask)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

    def generate_bs(self, encoder_outputs, encoder_hidden, target_tensor=None, mask=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = encoder_outputs.new_empty(batch_size, 1, dtype=torch.long).fill_(self.sos_idx)
        decoder_hidden = encoder_hidden

        decoder_output, decoder_hidden, _ = self.forward_step(
            decoder_input, decoder_hidden, encoder_outputs, mask=mask,
        )
        if target_tensor is None:
            return decoder_output.squeeze(1)

        for i in range(target_tensor.size(1)):
            decoder_input = target_tensor[:, i].unsqueeze(1)
            decoder_output, decoder_hidden, _ = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, mask=mask,
            )

        return decoder_output.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_tensor, target_tensor=None, src_padding_mask=None, tgt_padding_mask=None):
        # input_tensor  [b, t1]
        # target_tensor [b, t2]
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        # [b, t1, h]

        decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target_tensor, mask=src_padding_mask)
        # [b, t2, c]

        return decoder_outputs


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout: float,
        maxlen: int = 300,
    ):
        super().__init__()

        den = torch.exp(-torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))  # [t, d]
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)  # [1, t, d]

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1)])


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        emb_size: int,
        emb_src,
        emb_tgt,
        n_tokens,
        max_len,
        sos_idx,
        pos_max_len=300,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.max_len = max_len
        self.sos_idx = sos_idx
        self.src_tok_emb = emb_src
        self.tgt_tok_emb = emb_tgt
        self.semb_size = math.sqrt(emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size,
            dropout=dropout,
            maxlen=pos_max_len,
        )

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.generator = nn.Linear(emb_size, n_tokens)

    def forward(
        self,
        src,
        tgt=None,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        if tgt is None:
            return self.decode_forward(src, src_padding_mask=src_padding_mask)

        src_emb = self.src_embed(src)

        tgt = torch.cat(
            [
                src_emb.new_empty(src_emb.size(0), 1, dtype=torch.long).fill_(self.sos_idx),
                tgt[:, :-1],
            ],
            dim=1,
        )
        tgt_emb, tgt_mask = self.tgt_embed(tgt)

        memory_key_padding_mask = src_padding_mask

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        out = self.generator(out)

        return out

    def src_embed(self, src):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * self.semb_size)

        return src_emb

    def tgt_embed(self, tgt):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * self.semb_size)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)

        return tgt_emb, tgt_mask

    def decode_forward(self, src, src_padding_mask):
        memory = self.encode(src, src_key_padding_mask=src_padding_mask)

        decoder_input = memory.new_empty(memory.size(0), 1, dtype=torch.long).fill_(self.sos_idx)

        for _ in range(self.max_len):
            decoder_output = self.decode(
                decoder_input,
                memory,
                memory_key_padding_mask=src_padding_mask,
            )

            # Without teacher forcing: use its own predictions as the next input
            _, topi = decoder_output[:, -1].topk(1)
            decoder_input = torch.cat(
                [
                    decoder_input,
                    topi.detach(),  # detach from history as input
                ],
                dim=1,
            )

        return decoder_output

    def encode(self, src, src_key_padding_mask):
        src_emb = self.src_embed(src)

        out = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )

        return out

    def decode(self, tgt, memory, memory_key_padding_mask):
        tgt_emb, tgt_mask = self.tgt_embed(tgt)

        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        out = self.generator(out)

        return out


@torch.inference_mode()
def beam_search(
    model,
    X,
    mask,
    predictions=35,
    beam_width=4,
):
    """
    Implements Beam Search to compute the output with the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.

    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.

    predictions: int
        The number of tokens to append to X.

    beam_width: int
        The number of candidates to keep in the search.

    batch_size: int
        The batch size of the inner loop of the method, which relies on the beam width. 

    progress_bar: int 
        Shows a tqdm progress bar, useful for tracking progress with large tensors. Ranges from 0 to 2.

    Returns
    -------
    Y: LongTensor of shape (examples, length + predictions)
        The output sequences.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
        probability of the next token at every step.
    """

    encoder_outputs, encoder_hidden = model.encoder(X)
    # [b, t, d], ([2, b, d], [2, b, d])

    batch_size = encoder_outputs.size(0)

    # The next command can be a memory bottleneck, can be controlled with the batch 
    # size of the predict method.
    next_probabilities = model.decoder.generate_bs(encoder_outputs, encoder_hidden, mask=mask)  # [b, c]
    vocabulary_size = next_probabilities.shape[-1]
    probabilities, next_chars = next_probabilities.log_softmax(-1).topk(k=beam_width, dim=-1)  # [b, 4]
    Y = next_chars.reshape(-1, 1)  # [b * 4, 1]
    # This has to be minus one because we already produced a round
    # of predictions before the for loop.
    for _ in range(predictions - 1):
        dataset = torch.utils.data.TensorDataset(
            torch.repeat_interleave(encoder_outputs, beam_width, dim=0),
            torch.repeat_interleave(encoder_hidden[0], beam_width, dim=1).transpose(0, 1),
            torch.repeat_interleave(encoder_hidden[1], beam_width, dim=1).transpose(0, 1),
            torch.repeat_interleave(mask, beam_width, dim=0),
            Y,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        next_probabilities = []
        iterator = iter(loader)
        for x, e0, e1, m, y in iterator:
            e0 = e0.transpose(0, 1).contiguous()  # [b, 2, d] -> [2, b, d]
            e1 = e1.transpose(0, 1).contiguous()  # [b, 2, d] -> [2, b, d]
            e0e1 = e0, e1
            next_probabilities.append(model.decoder.generate_bs(x, e0e1, y, mask=m).log_softmax(-1))

        next_probabilities = torch.cat(next_probabilities, dim=0)
        next_probabilities = next_probabilities.reshape((-1, beam_width, next_probabilities.shape[-1]))
        probabilities = probabilities.unsqueeze(-1) + next_probabilities
        probabilities = probabilities.flatten(start_dim=1)
        probabilities, idx = probabilities.topk(k=beam_width, dim=-1)
        next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
        best_candidates = (idx / vocabulary_size).long()
        best_candidates += torch.arange(Y.shape[0] // beam_width, device=encoder_outputs.device).unsqueeze(-1) * beam_width
        Y = Y[best_candidates].flatten(end_dim=-2)
        Y = torch.cat([Y, next_chars], dim=1)

    return Y.reshape(-1, beam_width, Y.shape[-1]), probabilities
