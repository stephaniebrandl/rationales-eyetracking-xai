from torch import nn

gelu = nn.functional.gelu
import torch
import math
import torch.nn as nn
from torch.nn import functional as F


class Config(object):
    # LRP Config
    def __init__(self, model_name, device='cpu'):
        self.layer_norm_eps = 1e-12
        self.n_classes = 2
        if 'large' in model_name:
            self.n_blocks = 24
            self.num_attention_heads = 16
            self.hidden_size = 1024

        elif 'distilbert-base' in model_name:
            self.n_blocks = 6
            self.num_attention_heads = 12
            self.hidden_size = 768


        else:
            self.n_blocks = 12
            self.num_attention_heads = 12
            self.hidden_size = 768

        if 'xlm-roberta' in model_name:
            self.layer_norm_eps = 1e-5

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.detach_layernorm = True  # Detaches the attention-block-output LayerNorm
        self.detach_kq = True  # Detaches the kq-softmax branch
        self.detach_mean = False
        self.train_mode = False
        self.device = device


class ConfigGradienxInput(Config):
    def __init__(self, model_name, device='cpu'):
        super(ConfigGradienxInput, self).__init__(model_name, device)
        self.detach_layernorm = False  # Detaches the attention-block-output LayerNorm
        self.detach_kq = False  # Detaches the kq-softmax branch


class LayerNormImpl(nn.Module):
    __constants__ = ['weight', 'bias', 'eps']

    def __init__(self, args, hidden, eps=1e-6, elementwise_affine=True):
        super(LayerNormImpl, self).__init__()
        self.mode = args.lnv
        self.sigma = args.sigma
        self.hidden = hidden
        self.adanorm_scale = args.adanorm_scale
        self.nowb_scale = args.nowb_scale
        self.mean_detach = args.mean_detach
        self.std_detach = args.std_detach
        if self.mode == 'no_norm':
            elementwise_affine = False
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(hidden))
            self.bias = nn.Parameter(torch.Tensor(hidden))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.mode == 'no_norm':
            return input
        elif self.mode == 'topk':
            T, B, C = input.size()
            input = input.reshape(T * B, C)
            k = max(int(self.hidden * self.sigma), 1)
            input = input.view(1, -1, self.hidden)
            topk_value, topk_index = input.topk(k, dim=-1)
            topk_min_value, top_min_index = input.topk(k, dim=-1, largest=False)
            top_value = topk_value[:, :, -1:]
            top_min_value = topk_min_value[:, :, -1:]
            d0 = torch.arange(top_value.shape[0], dtype=torch.int64)[:, None, None]
            d1 = torch.arange(top_value.shape[1], dtype=torch.int64)[None, :, None]
            input[d0, d1, topk_index] = top_value
            input[d0, d1, top_min_index] = top_min_value
            input = input.reshape(T, B, self.hidden)
            return F.layer_norm(
                input, torch.Size([self.hidden]), self.weight, self.bias, self.eps)
        elif self.mode == 'nowb':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)

            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()

            input_norm = (input - mean) / (std + self.eps)
            return input_norm

        elif self.mode == 'layernorm':
            mean = input.mean(dim=-1, keepdim=True)
            # std = input.std(dim=-1, keepdim=True)
            # more robust implementation of std
            std = (torch.sum((input - mean) ** 2, dim=-1, keepdim=True) / (input.shape[-1])).sqrt()

            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()

            input_norm = (input - mean) / (std + self.eps)
            input_norm = input_norm * self.weight + self.bias
            return input_norm

        elif self.mode == 'gradnorm':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input - mean) / (std + self.eps)
            output = input.detach() + input_norm
            return output


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False, args=None):
    if args is not None:
        if args.lnv != 'origin':
            return LayerNormImpl(args, normalized_shape, eps, elementwise_affine)
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class LNargsDetachNotMean(object):

    def __init__(self):
        self.lnv = 'layernorm'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = True


class LNargsNoDetaching(object):

    def __init__(self):
        self.lnv = 'layernorm'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = False


class AttentionBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        ## attention
        self.query = nn.Linear(config.hidden_size, config.all_head_size)
        self.key = nn.Linear(config.hidden_size, config.all_head_size)
        self.value = nn.Linear(config.hidden_size, config.all_head_size)
        self.out_lin = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True)

        # Assume that we detach
        if self.config.train_mode == True:
            largs = LNargsNoDetaching()
        elif self.config.detach_layernorm == False and self.config.detach_mean == False:
            largs = LNargsNoDetaching()
        else:
            largs = LNargsDetachNotMean()
        self.sa_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, args=largs)

        ## FFN
        self.lin1 = nn.Linear(in_features=config.hidden_size, out_features=4 * config.hidden_size, bias=True)

        self.lin2 = nn.Linear(in_features=4 * config.hidden_size, out_features=config.hidden_size, bias=True)

        self.output_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, args=largs)

        self.detach = False

    def transpose_for_scores(self, x):
        # x torch.Size([1, 10, config.hidden_size])
        # xout torch.Size([1, 10, 12, 64])        
        new_x_shape = x.size()[:-1] + (self.config.num_attention_heads, self.config.attention_head_size)
        x = x.view(*new_x_shape)
        X = x.permute(0, 2, 1, 3)
        return X

    def forward(self, hidden_states, mask=None):

        def shape(x):
            """ separate heads """
            return x.view(1, -1, 12, 64).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(1, -1, 12 * 64)

        bs = hidden_states.shape[0]

        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        q = self.transpose_for_scores(self.query(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.config.attention_head_size)

        # Assume that we detach (NOT During Training!)
        if self.config.train_mode == True:
            weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        elif self.config.detach_layernorm == False and self.config.detach_mean == False:
            weights = nn.Softmax(dim=-1)(scores)
        else:
            weights = nn.Softmax(dim=-1)(scores).detach()

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)

        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context.size()[:-2] + (self.config.all_head_size,)
        context = context.view(new_context_layer_shape)

        # BERT SA Output
        sa_output = self.out_lin(context)
        sa_output = self.sa_layer_norm(sa_output + hidden_states)

        # Intermediate + output
        x = self.lin1(sa_output)
        identity = nn.Identity()
        x = identity(x) * (torch.nn.functional.gelu(x) / (identity(x) + 1e-9)).detach()

        # Layer output
        ffn_output = self.lin2(x)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)

        return ffn_output, weights


class TransformerExplainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_explainable = True

    @staticmethod
    def tanh_trick(x):
        identity = nn.Identity()
        tanh = nn.Tanh()
        x = identity(x) * (tanh(x) / (identity(x) + 1e-9)).detach()
        return x

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                labels=None,
                past_key_values_length=0,
                ):
        return NotImplemented

    def forward_and_explain(self, input_ids,
                            cl,
                            attention_mask=None,
                            token_type_ids=None,
                            position_ids=None,
                            inputs_embeds=None,
                            labels=None,
                            past_key_values_length=0,
                            device='cuda' if torch.cuda.is_available() else 'cpu'):
        return NotImplemented


class BertForQuestionAnsweringExplainer(TransformerExplainer):
    def __init__(self, config, pretrained_embeddings):
        super().__init__()

        self.n_blocks = config.n_blocks
        self.embeddings = pretrained_embeddings

        self.config = config
        self.attention_layers = torch.nn.Sequential(*[AttentionBlock(config) for i in range(self.n_blocks)])

        #  self.pooler =  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True)
        self.qa_outputs = nn.Linear(in_features=config.hidden_size, out_features=config.n_classes, bias=True)

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                labels=None,
                past_key_values_length=0,
                ):

        hidden_states = self.embeddings(input_ids=input_ids).to(self.config.device)

        attn_input = hidden_states
        for i, block in enumerate(self.attention_layers):
            output, attention_probs = block(attn_input)
            attn_input = output

        # pooled_output = output[:, 0]  # (bs, dim)

        # No pooling for qa!
        pooled_output = output

        #   pooled_output = self.pooler(pooled_output)  # (bs, dim)
        #   pooled_output = nn.Tanh()(pooled_output)  # (bs, dim)

        logits = self.qa_outputs(pooled_output)

        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        else:
            loss = None

        return {'loss': loss, 'logits': logits}

    def forward_and_explain(self, input_ids,
                            cl,
                            attention_mask=None,
                            token_type_ids=None,
                            position_ids=None,
                            inputs_embeds=None,
                            labels=None,
                            past_key_values_length=0,
                            device='cuda' if torch.cuda.is_available() else 'cpu'):

        # Forward
        As = {}

        hidden_states = self.embeddings(input_ids=input_ids).to(device)  # cuda()
        As['hidden_states'] = hidden_states
        attn_input = hidden_states

        for i, block in enumerate(self.attention_layers):
            # [1, 12, config.hidden_size] -> [1, 12, config.hidden_size]
            As['attn_input_{}'.format(i)] = attn_input
            output, attention_probs = block(As['attn_input_{}'.format(i)])
            attn_input = output

        # (1, 12, config.hidden_size) -> (1xconfig.hidden_size)

        As['output_all'] = output

        if False:
            output = output[:, 0]
            As['output'] = output

            pooled = nn.Tanh()(self.pooler(output))  # (bs, dim)         
            As['pooled'] = pooled

        else:
            output = output
            As['output'] = output
            pooled = output

        logits = self.qa_outputs(pooled)

        self.final_output = logits.detach().cpu().numpy()

        As['logits'] = logits
        self.probs = nn.Softmax(dim=-1)(As['logits'])

        #### Backward ####
        eps = 0.

        nlayers = len(self.attention_layers)

        layers = ['logits', 'output'] + ['attn_input_{}'.format(i) for i in range(nlayers)][::-1]
        layers_pre = ['output'] + ['attn_input_{}'.format(i) for i in range(nlayers)][::-1]

        Rs = {l: None for l in layers}

        logits_ = As['logits']
        logits_[:, cl != 1, 0] = 0.
        logits_[:, cl != 2, 1] = 0.

        Rs['logits'] = logits_  # As['logits'][:,cl]
        self.logit = Rs['logits']

        self.rattns = {}

        layer_dict = {'output': self.qa_outputs, }
        layer_dict.update({'attn_input_{}'.format(i): self.attention_layers[i] for i in range(nlayers)})

        for l, lplus in zip(layers_pre, layers):

            A = (As[l].data).requires_grad_(True)
            A0 = A

            layer = layer_dict[l]

            if l == 'pooled':
                raise
            # z = layer.forward(A)[:,cl] + eps

            # l: 'output', 'lplus': pooled
            elif l == 'output':
                z = layer.forward(A) + eps

                # l: 'attn_input_11', 'lplus': output
            elif 'attn_input' in l:
                idx = int(l.split('_')[-1])
                z = layer.forward(A)[0] + eps  # if idx==11 else layer.forward(A,  gamma=gamma)[0]+eps

            s = (Rs[lplus] / (z + 1e-9)).data
            (z * s).sum().backward();
            c = A.grad
            Rs[l] = (A * c).data

            if 'attn_input' in l:
                self.rattns[int(l.split('_')[-1])] = Rs[l].sum()

        R = Rs[l].sum(2)
        self.Rs = Rs

        return {'loss': None, 'logits': logits, 'R': R}

    @staticmethod
    def match_state_dicts(state_dict_src):
        renamed_state_dict = {}
        for k, v in state_dict_src.items():
            k_new = k.replace('roberta.', '').replace('bert.', '')
            if 'encoder.layer.' in k_new:
                k_new = k_new.replace('encoder.layer.', 'attention_layers.')

                k_new = k_new.replace('attention.output.dense.', 'out_lin.')
                k_new = k_new.replace('attention.output.LayerNorm.', 'sa_layer_norm.')
                k_new = k_new.replace('attention.', '')

                k_new = k_new.replace('intermediate.', 'lin1.')

                k_new = k_new.replace('output.dense.', 'lin2.')
                k_new = k_new.replace('output.LayerNorm.', 'output_layer_norm.')

                k_new = k_new.replace('self.', '')

            k_new = k_new.replace('model.', '')
            k_new = k_new.replace('dense.', '')
            renamed_state_dict[k_new] = v
        return renamed_state_dict


class DistilbertForQuestionAnsweringExplainer(BertForQuestionAnsweringExplainer):
    def __init__(self, config, pretrained_embeddings):
        super().__init__(config, pretrained_embeddings)

    @staticmethod
    def match_state_dicts(state_dict_src):
        renamed_state_dict = {}
        for k, v in state_dict_src.items():
            k_new = k.replace('distilbert.', '')
            if 'transformer.layer.' in k_new:
                k_new = k_new.replace('transformer.layer.', 'attention_layers.')

                k_new = k_new.replace('k_lin.', 'key.')
                k_new = k_new.replace('q_lin.', 'query.')
                k_new = k_new.replace('v_lin.', 'value.')
                k_new = k_new.replace('attention.output.dense.', 'out_lin.')

                k_new = k_new.replace('attention.output.LayerNorm.', 'sa_layer_norm.')
                k_new = k_new.replace('attention.', '')

                k_new = k_new.replace('ffn.lin1.', 'lin1.')

                k_new = k_new.replace('ffn.lin2.', 'lin2.')
            #   k_new = k_new.replace('output.LayerNorm.', 'output_layer_norm.')

            #  k_new = k_new.replace('self.', '')

            k_new = k_new.replace('model.', '')
            #   k_new = k_new.replace('dense.', '')
            renamed_state_dict[k_new] = v
        return renamed_state_dict
