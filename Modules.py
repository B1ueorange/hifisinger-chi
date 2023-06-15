from typing import List
import torch
import math
from argparse import Namespace  # for type
import logging
import numpy as np
from GST import STL, ReferenceEncoder

from nvlabs.torch_utils.ops import conv2d_gradfix
from nvlabs.torch_utils.ops.conv2d_gradfix import conv2d as nvlabs_conv2d, no_weight_gradients

conv2d_gradfix.enabled = True

class Sinusoidal_Positional_Embedding(torch.nn.Module): # 正弦位置嵌入？

    def __init__(self, channels, dropout=0.1, max_len=5000):
        super(Sinusoidal_Positional_Embedding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) #这里应该列是向量维度
        # pe = pe.unsqueeze(0).transpose(2, 1)  #[Batch, Channels, Time] unsqueeze 是扩维，transpose是交换坐标轴
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) #将pe标记为常量

    def forward(self, x):
        # x = x + self.pe[:, :, :x.size(2)]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Conv2d(torch.nn.Conv2d):
    def __init__(self, w_init_gain= 'relu', clamp: float=None, *args, **kwargs):
        self.w_init_gain = w_init_gain
        self.clamp = clamp #好像是一个阈值的限制，给定区间外的数都化成区间边界

        super().__init__(*args, **kwargs)
        self.runtime_Coef = 1.0 / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std= 1.0) # 给权重初始化，符合正态分布
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        x = nvlabs_conv2d(
            input= x,
            weight= self.weight.to(x.device) * self.runtime_Coef,
            stride= self.stride,
            padding= (int((self.kernel_size[0] - self.stride[0]) / 2), int((self.kernel_size[1] - self.stride[0]) / 2))
            )   # [Batch, Out, Resolution, Resolution]

        if not self.bias is None:
            x += self.bias.to(x.device)[None, :, None, None]

        if not self.clamp is None:
            x.clamp_(-self.clamp, self.clamp)

        return x

class Conv1d(Conv2d):

    def __init__(self, w_init_gain= 'relu', clamp: float=None, *args, **kwargs):
        kwargs['kernel_size'] = (1, kwargs['kernel_size'])
        super().__init__(w_init_gain, clamp, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        return super().forward(x.unsqueeze(2)).squeeze(2)



class Gradient_Penalty(torch.nn.Module): #我看不懂怎么惩罚的
    def __init__(
        self,
        gamma: float= 10.0
        ) -> None:
        super().__init__()

        self.gamma = gamma

    def forward(
        self,
        reals: torch.Tensor,
        discriminations: torch.Tensor,
        ) -> torch.Tensor:
        '''
        reals: [Batch, Channels, Time]. Real mels.
        discriminations: [Batch]. Discrimination outputs of real mels.
        '''
        with no_weight_gradients():
            gradient_Penalties = torch.autograd.grad(
                outputs= discriminations.sum(),
                inputs= reals,
                create_graph= True,
                only_inputs= True
                )[0]
            # output是求导因变量，input自变量，creategraph计算高阶导数
        gradient_Penalties = gradient_Penalties.square().sum(dim= (1, 2)) * (self.gamma * 0.5)   # [Batch]
        gradient_Penalties = (gradient_Penalties + reals[:, 0, 0] * 0.0).mean()
        # gradients = gradient.view(gradient.size(0), -1)
        # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_Penalties


class FFT_Block(torch.nn.Module): #这里应该只有encoder部分
    def __init__(
            self,
            in_channels: int,
            heads: int,
            dropout_rate: float,
            ff_in_kernel_size: int,
            ff_out_kernel_size: int,
            ff_channels: int
    ):
        super(FFT_Block, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Multihead_Attention'] = torch.nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=heads
        )
        self.layer_Dict['LayerNorm_0'] = torch.nn.LayerNorm(
            normalized_shape=in_channels
        )
        self.layer_Dict['Dropout'] = torch.nn.Dropout(p=dropout_rate)
        self.layer_Dict['Conv'] = torch.nn.Sequential()
        self.layer_Dict['Conv'].add_module('Conv_0', Conv1d(
            in_channels=in_channels,
            out_channels=ff_channels,
            kernel_size=ff_in_kernel_size,
            padding=(ff_in_kernel_size - 1) // 2,
            w_init_gain='relu'
        ))
        self.layer_Dict['Conv'].add_module('ReLU', torch.nn.ReLU())
        self.layer_Dict['Conv'].add_module('Conv_1', Conv1d(
            in_channels=ff_channels,
            out_channels=in_channels,
            kernel_size=ff_out_kernel_size,
            padding=(ff_out_kernel_size - 1) // 2,
            w_init_gain='linear'
        ))
        self.layer_Dict['Conv'].add_module('Dropout', torch.nn.Dropout(p=dropout_rate))
        self.layer_Dict['LayerNorm_1'] = torch.nn.LayerNorm(
            normalized_shape=in_channels
        )

    def forward(self, x: torch.FloatTensor, masks: torch.BoolTensor = None):
        '''
        x: [Batch, Channels, Time]
        '''
        x = self.layer_Dict['Multihead_Attention'](
            query=x.permute(2, 0, 1), #交换维度，比如说现在2交换到原来0的位置，0到1，1到2
            key=x.permute(2, 0, 1),
            value=x.permute(2, 0, 1),
            key_padding_mask=masks
        )[0].permute(1, 2, 0) + x
        x = self.layer_Dict['LayerNorm_0'](x.transpose(2, 1)).transpose(2, 1)
        x = self.layer_Dict['Dropout'](x)
        if not masks is None:
            x *= torch.logical_not(masks).unsqueeze(1).float()
        x = self.layer_Dict['Conv'](x) + x
        x = self.layer_Dict['LayerNorm_1'](x.transpose(2, 1)).transpose(2, 1)

        if not masks is None:
            x *= torch.logical_not(masks).unsqueeze(1).float()

        return x

class Duration_Predictor(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super(Duration_Predictor, self).__init__()
        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()

        previous_Channels = self.hp.Encoder.Size
        for index, (kernel_Size, channels) in enumerate(zip(
            self.hp.Duration_Predictor.Conv.Kernel_Size,
            self.hp.Duration_Predictor.Conv.Channels
            )): #enumerate自带序号，然后后面那两维度用zip封了
            self.layer_Dict['Conv_{}'.format(index)] = Conv1d(
                in_channels= previous_Channels,
                out_channels= channels,
                kernel_size= kernel_Size,
                padding= (kernel_Size - 1) // 2,
                w_init_gain= 'relu'
                )
            self.layer_Dict['LayerNorm_{}'.format(index)] = torch.nn.LayerNorm(
                normalized_shape= channels
                )
            self.layer_Dict['ReLU_{}'.format(index)] = torch.nn.ReLU()
            self.layer_Dict['Dropout_{}'.format(index)] = torch.nn.Dropout(
                p= self.hp.Duration_Predictor.Conv.Dropout_Rate
                )
            previous_Channels = channels

        self.layer_Dict['Projection'] = torch.nn.Sequential()
        self.layer_Dict['Projection'].add_module('Conv', Conv1d(
            in_channels= previous_Channels,
            out_channels= 1,
            kernel_size= 1,
            w_init_gain= 'relu'
            ))
        self.layer_Dict['Projection'].add_module('ReLU', torch.nn.ReLU())

    def forward(
        self,
        encodings: torch.FloatTensor,
        durations: torch.LongTensor= None
        ):
        x = encodings
        for index in range(len(self.hp.Duration_Predictor.Conv.Kernel_Size)):
            x = self.layer_Dict['Conv_{}'.format(index)](x)
            x = self.layer_Dict['LayerNorm_{}'.format(index)](x.transpose(2, 1)).transpose(2, 1)
            x = self.layer_Dict['ReLU_{}'.format(index)](x)
            x = self.layer_Dict['Dropout_{}'.format(index)](x)
        predicted_Durations = self.layer_Dict['Projection'](x)

        if durations is None: # 为什么会出现预测不出duration的情况
            durations = predicted_Durations.ceil().long().clamp(0, self.hp.Max_Duration) # 保证durations在给定的数据范围内
            durations = torch.stack([
                (torch.ones_like(duration) if duration.sum() == 0 else duration)
                for duration in durations
                ], dim= 0) # 语法问题

            max_Durations = torch.max(torch.cat([duration.sum(dim= 0, keepdim= True) + 1 for duration in durations]))
            if max_Durations > self.hp.Max_Duration:  # I assume this means failing
                durations = torch.ones_like(predicted_Durations).long()
            else:
                durations = torch.cat([
                    durations[:, :-1], durations[:, -1:] + max_Durations - durations.sum(dim= 1, keepdim= True)
                    ], dim= 1) # 向量加上一个数是？？？

        x = torch.stack([
            encoding.repeat_interleave(duration, dim= 1)
            for encoding, duration in zip(encodings, durations)
            ], dim= 0) # 按照计算好的持续时间扩展

        return x, predicted_Durations.squeeze(1)

class ScaledDotProductAttention(torch.nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(torch.nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = torch.nn.Linear(d_model, n_head * d_k)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = torch.nn.LayerNorm(d_model)

        self.fc = torch.nn.Linear(n_head * d_v, d_model)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(torch.nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = torch.nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = torch.nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = torch.nn.LayerNorm(d_in)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(torch.nn.functional.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output

class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output

# class Encoder(torch.nn.Module):
#
#     def __init__(self, hyper_parameters: Namespace):
#
#         super(Encoder, self).__init__()
#         self.hp = hyper_parameters
#
#         self.layer_Dict = torch.nn.ModuleDict()
#         self.layer_Dict['Phoneme_Embedding'] = torch.nn.Embedding( #这里应该都是初始化向量
#             num_embeddings=self.hp.Tokens,
#             embedding_dim=self.hp.Encoder.Size,
#         )
#         self.layer_Dict['Duration_Embedding'] = torch.nn.Embedding(
#             num_embeddings=self.hp.Max_Duration,
#             embedding_dim=self.hp.Encoder.Size,
#         )
#         self.layer_Dict['Note_Embedding'] = torch.nn.Embedding(
#             num_embeddings=self.hp.Max_Note,
#             embedding_dim=self.hp.Encoder.Size,
#         )
#
#         self.layer_Dict['Positional_Embedding'] = Sinusoidal_Positional_Embedding(
#             channels=self.hp.Encoder.Size,
#             dropout=0.0
#         )
#
#         for index in range(self.hp.Encoder.FFT_Block.Stacks):
#             self.layer_Dict['FFT_Block_{}'.format(index)] = FFT_Block( # 构造transformer网络
#                 in_channels=self.hp.Encoder.Size,
#                 heads=self.hp.Encoder.FFT_Block.Heads,
#                 dropout_rate=self.hp.Encoder.FFT_Block.Dropout_Rate,
#                 ff_in_kernel_size=self.hp.Encoder.FFT_Block.FeedForward.In_Kernel_Size,
#                 ff_out_kernel_size=self.hp.Encoder.FFT_Block.FeedForward.Out_Kernel_Size,
#                 ff_channels=self.hp.Encoder.FFT_Block.FeedForward.Channels,
#             )
#
#     def forward(
#             self,
#             tokens: torch.LongTensor,
#             durations: torch.LongTensor,
#             notes: torch.LongTensor,
#             masks: torch.BoolTensor = None
#     ):
#         '''
#         x: [Batch, Time]
#         lengths: [Batch]
#         '''
#         tokens = self.layer_Dict['Phoneme_Embedding'](tokens).transpose(2, 1)  # [Batch, Channels, Time]
#         durations = self.layer_Dict['Duration_Embedding'](durations).transpose(2, 1)  # [Batch, Channels, Time]
#         notes = self.layer_Dict['Note_Embedding'](notes).transpose(2, 1)  # [Batch, Channels, Time]
#
#         x = self.layer_Dict['Positional_Embedding'](tokens + durations + notes)
#         for index in range(self.hp.Encoder.FFT_Block.Stacks):
#             x = self.layer_Dict['FFT_Block_{}'.format(index)](x, masks)
#
#         return x  # [Batch, Channels, Time]

class Encoder(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super(Encoder, self).__init__()
        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Phoneme_Embedding'] = torch.nn.Embedding(
            num_embeddings=self.hp.Tokens,
            embedding_dim=self.hp.Encoder.Size,
        )
        self.layer_Dict['Duration_Embedding'] = torch.nn.Embedding(
            num_embeddings=self.hp.Max_Duration,
            embedding_dim=self.hp.Encoder.Size,
        )
        self.layer_Dict['Note_Embedding'] = torch.nn.Embedding(
            num_embeddings=self.hp.Max_Note,
            embedding_dim=self.hp.Encoder.Size,
        )

        self.layer_Dict['Positional_Embedding'] = Sinusoidal_Positional_Embedding(
            channels=self.hp.Encoder.Size,
            dropout=0.0
        )

        for index in range(self.hp.Encoder.FFT_Block.Stacks):
            # self.layer_Dict['FFT_Block_{}'.format(index)] = FFT_Block(
            #     in_channels= self.hp.Encoder.Size,
            #     heads= self.hp.Encoder.FFT_Block.Heads,
            #     dropout_rate= self.hp.Encoder.FFT_Block.Dropout_Rate,
            #     ff_in_kernel_size= self.hp.Encoder.FFT_Block.FeedForward.In_Kernel_Size,
            #     ff_out_kernel_size= self.hp.Encoder.FFT_Block.FeedForward.Out_Kernel_Size,
            #     ff_channels= self.hp.Encoder.FFT_Block.FeedForward.Channels,
            #     )
            self.layer_Dict['FFT_Block_{}'.format(index)] = FFTBlock \
                    (
                    d_model=self.hp.Encoder.Size,
                    n_head=self.hp.Encoder.FFT_Block.Heads,
                    d_k=(self.hp.Encoder.Size // self.hp.Encoder.FFT_Block.Heads),
                    d_v=(self.hp.Encoder.Size // self.hp.Encoder.FFT_Block.Heads),
                    d_inner=1024,
                    kernel_size=[9, 1],
                    dropout=0.2,
                )

    def get_mask_from_lengths(self, lg, max_len=None):
        batch_size = lg.shape[0]
        if max_len is None:
            max_len = torch.max(lg).item()

        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
        mask = ids >= lg.expand(-1, max_len)

        return mask

    def forward(
            self,
            tokens: torch.LongTensor,
            durations: torch.LongTensor,
            notes: torch.LongTensor,
            masks: torch.BoolTensor = None
    ):
        '''
        x: [Batch, Time]
        lengths: [Batch]
        '''
        slf_attn_mask = masks.unsqueeze(1).expand(-1, tokens.size(1), -1)
        tokens = self.layer_Dict['Phoneme_Embedding'](tokens)
        durations = self.layer_Dict['Duration_Embedding'](durations)
        notes = self.layer_Dict['Note_Embedding'](notes)
        x = self.layer_Dict['Positional_Embedding'](tokens + durations + notes)
        # tokens = self.layer_Dict['Phoneme_Embedding'](tokens).transpose(2, 1)  # [Batch, Channels, Time]
        # durations = self.layer_Dict['Duration_Embedding'](durations).transpose(2, 1)  # [Batch, Channels, Time]
        # notes = self.layer_Dict['Note_Embedding'](notes).transpose(2, 1)  # [Batch, Channels, Time]
        # x = self.layer_Dict['Positional_Embedding'](tokens + durations + notes)
        # x = x.transpose(2, 1)
        # mm = torch.logical_not(masks).unsqueeze(1).float()
        # mm = self.get_mask_from_lengths(tokens, tokens.size(1))
        for index in range(self.hp.Encoder.FFT_Block.Stacks):
            x = self.layer_Dict['FFT_Block_{}'.format(index)](x, masks, slf_attn_mask)

        return x  # [Batch, Channels, Time]

class Decoder(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super(Decoder, self).__init__()
        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Positional_Embedding'] = Sinusoidal_Positional_Embedding(
            channels= self.hp.Encoder.Size
            )

        self.layer_Dict['FFT_Block'] = torch.nn.Sequential()
        for index in range(self.hp.Decoder.FFT_Block.Stacks):
            # self.layer_Dict['FFT_Block_{}'.format(index)] = FFT_Block(
            #     in_channels= self.hp.Encoder.Size,
            #     heads= self.hp.Decoder.FFT_Block.Heads,
            #     dropout_rate= self.hp.Decoder.FFT_Block.Dropout_Rate,
            #     ff_in_kernel_size= self.hp.Decoder.FFT_Block.FeedForward.In_Kernel_Size,
            #     ff_out_kernel_size= self.hp.Decoder.FFT_Block.FeedForward.Out_Kernel_Size,
            #     ff_channels= self.hp.Decoder.FFT_Block.FeedForward.Channels,
            #     )
            self.layer_Dict['FFT_Block_{}'.format(index)] = FFTBlock \
                    (
                    d_model=self.hp.Decoder.Size,
                    n_head=self.hp.Decoder.FFT_Block.Heads,
                    d_k=(self.hp.Decoder.Size // self.hp.Decoder.FFT_Block.Heads),
                    d_v=(self.hp.Decoder.Size // self.hp.Decoder.FFT_Block.Heads),
                    d_inner=1024,
                    kernel_size=[9, 1],
                    dropout=0.2,
                )

        self.layer_Dict['Projection'] = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Sound.Mel_Dim + 1 + 1,
            kernel_size= 1,
            w_init_gain= 'linear'
            )

    def forward(
        self,
        encodings: torch.FloatTensor,
        masks: torch.BoolTensor
        ):
        x = encodings
        x = x.transpose(2, 1)
        x = self.layer_Dict['Positional_Embedding'](x)
        slf_attn_mask = masks.unsqueeze(1).expand(-1, x.size(1), -1)
        for index in range(self.hp.Encoder.FFT_Block.Stacks):
            # x = self.layer_Dict['FFT_Block_{}'.format(index)](x, masks= masks)
            x = self.layer_Dict['FFT_Block_{}'.format(index)](x, masks, slf_attn_mask)
        x = x.transpose(2, 1)
        x = self.layer_Dict['Projection'](x)

        mels, silences, notes = torch.split(
            x,
            split_size_or_sections= [self.hp.Sound.Mel_Dim, 1, 1],
            dim= 1
            ) #也就是说，按照第二维分割

        return mels, silences.squeeze(1), notes.squeeze(1)

class Speaker_Embedding_Changer(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super(Speaker_Embedding_Changer, self).__init__()
        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['MLP_0'] = torch.nn.Linear(in_features=256, out_features=self.hp.Encoder.Size, bias=True)
        self.layer_Dict['Leaky_ReLU_0'] = torch.nn.LeakyReLU(
            negative_slope=0.2,
            inplace=True
        )

    def forward(self,
                encodings: torch.FloatTensor):
        x = encodings
        x = self.layer_Dict['MLP_0'](x)
        x = self.layer_Dict['Leaky_ReLU_0'](x)
        return x

class HifiSinger(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super(HifiSinger, self).__init__()

        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Encoder'] = Encoder(self.hp)
        self.layer_Dict['Duration_Predictor'] = Duration_Predictor(self.hp)
        self.layer_Dict['Speaker_Embedding_Changer'] = Speaker_Embedding_Changer(self.hp)
        # self.layer_Dict['Reference_Encoder'] = ReferenceEncoder(self.hp)
        self.layer_Dict['STL'] = STL(self.hp)
        # self.layer_Dict['Mlp'] = torch.nn.Linear(in_features=256, out_features=self.hp.Encoder.Size, bias=True)
        self.layer_Dict['Decoder'] = Decoder(self.hp)

    def forward(
            self,
            durations,
            tokens,
            notes,
            token_lengths=None,  # token_length == duration_length == note_length
            speaker_embeddings=None
    ):
        encoder_Masks = None
        if not token_lengths is None: # 为啥遮住了
            encoder_Masks = self.Mask_Generate(
                lengths=token_lengths,
                max_lengths=tokens.size(1)
            )

        encodings = self.layer_Dict['Encoder'](
            tokens=tokens,
            durations=durations,
            notes=notes,
            masks=encoder_Masks
        )

        encodings = encodings.transpose(2, 1)

        encodings, predicted_Durations = self.layer_Dict['Duration_Predictor'](
            encodings=encodings,
            durations=durations
        ) # 最后计算持续时间

        if(speaker_embeddings is not None):
            # token_numbers = torch.max(token_lengths)
            token_numbers = encodings.size(2)
            batch_size = encodings.size(0)
            speaker_embeddings = self.layer_Dict['STL'](speaker_embeddings)
            speaker_embeddings = speaker_embeddings.squeeze(1)
            speaker_embeddings = self.layer_Dict['Speaker_Embedding_Changer'](speaker_embeddings)
            tmp_embed = torch.FloatTensor()
            # tmp_embed = torch.stack([torch.stack([speaker_embeddings[i] for j in range(token_numbers)], dim=1) for i in range(self.hp.Train.Batch_Size)], dim=0)

            for i in range(batch_size): 
                tmp_embed2 = []
                for j in range(token_numbers):
                    tmp_embed2.append(speaker_embeddings[i])
                tmp_embed1 = torch.stack(tmp_embed2, dim=1)
                tmp_embed1 = tmp_embed1.unsqueeze(0)
                if(i == 0):
                    tmp_embed = tmp_embed1
                else:
                    tmp_embed = torch.cat((tmp_embed, tmp_embed1), 0)
            # for i in range(self.hp.Train.Batch_Size):
            #     tmp_embed = torch.stack((tmp_embed, torch.stack([speaker_embeddings[i] for j in range(token_numbers)], dim=1).unsqueeze(0)), dim=0)
                # tmp_embed = torch.stack((tmp_embed, torch.stack([speaker_embeddings[i] for j in range(token_numbers)], dim=1)), dim=0)
            encodings = encodings + tmp_embed

        decoder_Masks = self.Mask_Generate(
            lengths=durations[:, :-1].sum(dim=1),
            max_lengths=durations[0].sum()
        )
        # decoder_Masks = None

        predicted_Mels, predicted_Silences, predicted_Pitches = self.layer_Dict['Decoder'](
            encodings=encodings,
            masks=decoder_Masks
        )

        predicted_Pitches = predicted_Pitches + torch.stack([
            note.repeat_interleave(duration) / self.hp.Max_Note
            for note, duration in zip(notes, durations)
        ], dim=0)

        predicted_Mels.data.masked_fill_(decoder_Masks.unsqueeze(1), -self.hp.Sound.Max_Abs_Mel)
        predicted_Silences.data.masked_fill_(decoder_Masks, 0.0)  # 0.0 -> Silence, 1.0 -> Voice
        predicted_Pitches.data.masked_fill_(decoder_Masks, 0.0)

        # test_speaker_embeddings = self.layer_Dict['Reference_Encoder'](predicted_Mels.transpose(1, 2))

        return predicted_Mels, torch.sigmoid(predicted_Silences), predicted_Pitches, predicted_Durations

    def Mask_Generate(self, lengths, max_lengths=None):
        '''
        lengths: [Batch]
        '''
        sequence = torch.arange(max_lengths or torch.max(lengths))[None, :].to(lengths.device) # 语法问题还是，arange函数是构造一维向量，按照步长增加
        return sequence >= lengths[:, None]  # [Batch, Time]

class Discriminator(torch.nn.Module):
    def __init__(
        self,
        stacks: int,
        kernel_size: int,
        channels: int,
        frequency_range: List[int]
        ) -> None:
        super(Discriminator, self).__init__()

        self.frequency_Range = frequency_range

        self.layer = torch.nn.Sequential()

        self.stack_num = int(stacks)
        self.channel_num = channels
        self.layer_dict = torch.nn.ModuleDict()

        previous_Channels = 1
        for index in range(stacks):
            # self.layer_dict['Conv_{}'.format(index)] = torch.nn.Conv2d(
            #     in_channels=previous_Channels,
            #     out_channels=channels,
            #     kernel_size=kernel_size,
            #     bias=False
            # )
            self.layer_dict['Conv_{}'.format(index)] = Conv2d(
                in_channels= previous_Channels,
                out_channels= channels,
                kernel_size= kernel_size,
                bias= False,
                w_init_gain= 'linear'
                )
            self.layer_dict['Leaky_ReLU_{}'.format(index)] = torch.nn.LeakyReLU(
                negative_slope= 0.2,
                inplace= True
            )
            # self.layer.add_module('Conv_{}'.format(index), torch.nn.Conv2d(
            #     in_channels=previous_Channels,
            #     out_channels=channels,
            #     kernel_size=kernel_size,
            #     bias=False
            # ))
            # self.layer.add_module('Conv_{}'.format(index), Conv2d(
            #     in_channels= previous_Channels,
            #     out_channels= channels,
            #     kernel_size= kernel_size,
            #     bias= False,
            #     w_init_gain= 'linear'
            #     ))
            # self.layer.add_module('Leaky_ReLU_{}'.format(index), torch.nn.LeakyReLU(
            #     negative_slope= 0.2,
            #     inplace= True
            #     ))
            previous_Channels = channels

        # self.layer_dict['Projection'] = torch.nn.Conv2d(
        #     in_channels=previous_Channels,
        #     out_channels=1,
        #     kernel_size=1,
        #     bias=True
        # )

        self.layer_dict['Projection'] =  Conv2d(
            in_channels= previous_Channels,
            out_channels= 1,
            kernel_size= 1,
            bias= True,
            w_init_gain= 'linear'
            )
        # self.layer.add_module('Projection', torch.nn.Conv2d(
        #     in_channels=previous_Channels,
        #     out_channels=1,
        #     kernel_size=1,
        #     bias=False
        # ))
        # self.layer.add_module('Projection', torch.nn.Linear(
        #     previous_Channels,
        #     1,
        #     bias=True
        # ))
        # self.layer.add_module('Projection', Conv2d(
        #     in_channels= previous_Channels,
        #     out_channels= 1,
        #     kernel_size= 1,
        #     bias= False,
        #     w_init_gain= 'linear'
        #     ))



    def forward(
        self,
        x: torch.FloatTensor,
        lengths: torch.LongTensor
        ):
        '''
        x: [Batch, Mel_dim, Time]
        '''

        sampling_Length = lengths.min()
        mels = []
        for mel, length in zip(x, lengths):
            offset = torch.randint(
                low= 0,
                high= length - sampling_Length + 1,
                size= (1,)
                ).to(x.device)
            # print(offset)
            ttmp = mel[self.frequency_Range[0]:self.frequency_Range[1], offset:offset + sampling_Length]
            # print(ttmp)
            mels.append(ttmp)

        mels = torch.stack(mels).unsqueeze(dim= 1)    # [Batch, 1, Sampled_Dim, Min_Time])


        mels = self.layer_dict['Conv_0'](mels)
        mels = self.layer_dict['Leaky_ReLU_0'](mels)
        mels = self.layer_dict['Conv_1'](mels)
        mels = self.layer_dict['Leaky_ReLU_1'](mels)
        mels = self.layer_dict['Conv_2'](mels)
        mels = self.layer_dict['Leaky_ReLU_2'](mels)
        mels = self.layer_dict['Projection'](mels)
        return mels.squeeze(dim=1)
        # return self.layer(mels).squeeze(dim= 1) # [Batch, Sampled_Dim, Min_Time]


class Discriminators(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace) -> None:
        super(Discriminators, self).__init__()
        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()

        for index, frequency_Range in enumerate(self.hp.Discriminator.Frequency_Range):
            self.layer_Dict['Discriminator_{}'.format(index)] = Discriminator(
                stacks= self.hp.Discriminator.Stacks,
                channels= self.hp.Discriminator.Channels,
                kernel_size= self.hp.Discriminator.Kernel_Size,
                frequency_range= frequency_Range
                )

    def forward(
        self,
        x: torch.FloatTensor,
        lengths: torch.LongTensor
        ):
        '''
        x: [Batch, Time]
        '''
        # rt = []
        # for index in range(len(self.hp.Discriminator.Frequency_Range)):
        #     #self.layer_Dict['Discriminator_{}'.format(index)].setlength(length)
        #     print(index)
        #     rt.append(self.layer_Dict['Discriminator_{}'.format(index)](x, lengths))
        # return rt
        return [
            self.layer_Dict['Discriminator_{}'.format(index)](x, lengths)
            for index in range(len(self.hp.Discriminator.Frequency_Range))
            ]


if __name__ == "__main__":
    import yaml
    from Arg_Parser import Recursive_Parse

    hp = Recursive_Parse(yaml.load(
        open('Hyper_Parameters.yaml', encoding='utf-8'),
        Loader=yaml.Loader
    ))

    from Datasets import Dataset, Collater

    # token_Dict = yaml.load(open(hp.Token_Path), Loader=yaml.Loader)
    token_Dict = {}
    with open('./token1.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tmp1 = line.split()
            print(tmp1)
            token_Dict[tmp1[0]] = int(tmp1[2])
    token_Dict['AP'] = 59
    token_Dict['SP'] = 59
    token_Dict['<X>'] = 59
    dataset = Dataset(
        pattern_path=hp.Train.Train_Pattern.Path,
        Metadata_file=hp.Train.Train_Pattern.Metadata_File,
        token_dict=token_Dict,
        accumulated_dataset_epoch=hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
    )
    collater = Collater(
        token_dict=token_Dict,
        max_abs_mel=hp.Sound.Max_Abs_Mel
    )
    dataLoader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=collater,
        sampler=torch.utils.data.RandomSampler(dataset),
        batch_size=hp.Train.Batch_Size,
        num_workers=hp.Train.Num_Workers,
        pin_memory=True
    )

    durations, tokens, notes, token_lengths, mels, silences, pitches, mel_lengths = next(iter(dataLoader))

    model = HifiSinger(hp)
    predicted_Mels, predicted_Silences, predicted_Pitches, predicted_Durations = model(
        tokens=tokens,
        durations=durations,
        notes=notes,
        token_lengths=token_lengths
    )

    discriminator = Discriminators(hp)
    discriminations = discriminator(predicted_Mels, mel_lengths)

    print(discriminations)
    for x in discriminations:
        print(x.shape)