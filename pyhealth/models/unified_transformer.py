# Originally, adapted from https://github.com/ttumyche/UniXGen/blob/main/transformer_pytorch/transformer_unified.py
# However, it contained many Pythonic loops and wasn't a vectorized implementation causing bottlenecks,
# so was improvised with vectorized torch ops removing loops. Tested to run on CUDA and XLA (TPU) devices.


import math
import numpy as np
from functools import partial

from pyhealth.models.favor_attention.utils import *
from pyhealth.models.favor_attention import FAVORAttention, ProjectionUpdater, AxialPositionalEmbedding


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

VIEW_MAP = {
    'AP': 0,
    'PA': 1,
    'LATERAL': 2,
    'LL': 2,       # Map LL to same as LATERAL
    'PAD': 3
}

class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None):
        super().__init__()

        activation = default(activation, nn.GELU)

        self.w1 = nn.Linear(dim, mult * dim)
        self.act = activation()
        self.w2 = nn.Linear(mult * dim, dim)
        self.dropout = dropout

    def forward(self, x, **kwargs):
        out = self.w1(x)
        out = self.act(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.w2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            local_attn_heads=0,
            causal='conditioned_causal',
            attn_type="conditioned_noncuda",
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            ff_mult=4,
            nb_features=None,
            feature_redraw_interval=1000,
            use_scalenorm=False,
            use_rezero=False,
            ff_dropout=0.,
            attn_dropout=0.,
            cross_attend=False,
            auto_check_redraw=True,
            qkv_bias=True,
            attn_out_bias=True,
            no_projection=False,
            FAVOR=False,
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)  # eg. dim=512

        if FAVOR:
            for _, local_heads in zip(range(depth), local_attn_heads):
                layers.append(nn.ModuleList([
                    wrapper_fn(FAVORAttention(dim=dim, causal=causal, attn_type=attn_type,
                                              generalized_attention=generalized_attention,
                                              kernel_fn=kernel_fn,
                                              heads=heads, local_heads=local_heads, nb_features=nb_features,
                                              dropout=attn_dropout, no_projection=no_projection,
                                              qkv_bias=qkv_bias, attn_out_bias=attn_out_bias)),
                    wrapper_fn(PositionWiseFeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, activation=None))
                ]))
                if not cross_attend:
                    continue

        execute_type = SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)  # len(): 2*depth if cross_attend else depth
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'pad_mask': route_attn, 'pos_emb': route_attn, 'causal': route_attn, 'condition_len': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route={**attn_route_map, **context_route_map})

        if FAVOR:
            self.auto_check_redraw = auto_check_redraw  # auto_check_redraw = True
            self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)
        else:
            self.auto_check_redraw = False

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, **kwargs)


class UnifiedTransformerLM(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,  # text vocab size
            num_img_tokens,  # img vocab size + num img pad
            img_vocab_size,
            max_seq_len,  # total max len; img_len * max_img_num + max_text_len
            max_img_len,
            max_img_num,  # num img slot
            img_len,
            dim,
            depth,
            heads=8,
            local_attn_heads=0,
            causal='conditioned_causal',
            attn_type="conditioned_noncuda",
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            ff_mult=4,
            nb_features=None,
            feature_redraw_interval=1000,
            reversible=False,
            emb_dropout=0.,
            ff_dropout=0.,
            attn_dropout=0.,
            use_scalenorm=False,
            use_rezero=False,
            cross_attend=False,
            no_projection=False,
            tie_embed=False,
            rotary_position_emb=True,
            axial_position_emb=False,
            axial_position_shape=None,
            auto_check_redraw=True,
            qkv_bias=False,
            attn_out_bias=False,
            img_fmap_size=0,
            FAVOR=False,

            mask_prob=0.15,
            replace_prob=0.9,
            random_token_prob=0.,
            mask_token_id=4,
            pad_token_id=0,
            mask_ignore_token_ids=[],
            **kwargs
    ):
        super().__init__()

        self.img_len = img_len
        self.num_txt_tokens = num_tokens
        self.num_img_tokens = num_img_tokens
        self.img_vocab_size = img_vocab_size
        self.max_seq_len = max_seq_len
        self.max_img_num = max_img_num
        local_attn_heads = cast_tuple(local_attn_heads)
        self.dim = dim
        dim_head = dim // heads

        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_token_prob = random_token_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mask_ignore_token_ids = mask_ignore_token_ids

        self.attn_type = attn_type

        # !# img
        self.image_token_emb = nn.Embedding(num_img_tokens, dim)
        self.ap_att_emb = AbsolutePositionalEmbedding(dim, max_seq_len)  # max_seq_len = img_len * max_img_num + max_text_len
        self.pa_att_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.la_att_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.pad_img_att_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.image_pos_emb = AxialPositionalEmbedding(dim=dim, axial_shape=(img_fmap_size + 1, img_fmap_size + 1))

        # !# text
        self.token_emb = nn.Embedding(num_tokens, dim)
        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)
        self.txt_att_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, local_attn_heads, causal,
                                       attn_type, generalized_attention,
                                       kernel_fn,
                                       ff_mult, nb_features, feature_redraw_interval, use_scalenorm, use_rezero,
                                       ff_dropout, attn_dropout, cross_attend, auto_check_redraw,
                                       qkv_bias, attn_out_bias, no_projection, FAVOR)
        self.norm = nn.LayerNorm(dim)

        self.to_out_txt = nn.Linear(dim, num_tokens)  # if not tie_embed else None
        self.to_out_img = nn.Linear(dim, num_img_tokens)  # if not tie_embed else None
        self.to_out_combined_txt_img = nn.Linear(dim, (num_tokens + num_img_tokens))

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, batch, causal, return_encodings=False, **kwargs):  # kwargs = {'mask': tensor with same shape x}
        txt, view = batch['txt'], batch['view_position']
        b, n_txt, device = *txt.shape, txt.device

        img1 = batch['img1']
        b, n_img1, device = *img1.shape, img1.device
        img2 = batch['img2']
        b, n_img2, device = *img2.shape, img2.device
        img3 = batch['img3']
        b, n_img3, device = *img3.shape, img3.device

        n = n_img1 + n_txt + n_img2 + n_img3
        imgs = [img1, img2, img3]

        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # !# image; token and positional embeddings
        # -- VECTORIZED!

        # Step 1: Apply token embeddings to all images
        x_img1 = self.image_token_emb(img1)  # [b, img_len1, dim]
        x_img2 = self.image_token_emb(img2)  # [b, img_len2, dim]
        x_img3 = self.image_token_emb(img3)  # [b, img_len3, dim]

        # Step 2: Apply positional embeddings
        pos_img1 = self.image_pos_emb(x_img1)  # [b, img_len1, dim]
        pos_img2 = self.image_pos_emb(x_img2)  # [b, img_len2, dim]
        pos_img3 = self.image_pos_emb(x_img3)  # [b, img_len3, dim]

        # Step 3: Apply attention embeddings based on view types
        # Stack views for vectorized operations
        # views = torch.stack(view, dim=1)  # [b, 3]

        # Create embeddings for each view type
        # For image 1,2,3
        view_masks1 = F.one_hot(view[:, 0], num_classes=4).float()  # [b, 4]
        view_masks2 = F.one_hot(view[:, 1], num_classes=4).float()
        view_masks3 = F.one_hot(view[:, 2], num_classes=4).float()

        # Apply embeddings for different view types
        att_tensors1 = torch.stack([
            self.ap_att_emb(x_img1),
            self.pa_att_emb(x_img1),
            self.la_att_emb(x_img1),
            self.pad_img_att_emb(x_img1)
        ], dim=1)  # [b, 4, seq_len, dim]

        att_tensors2 = torch.stack([
            self.ap_att_emb(x_img2),
            self.pa_att_emb(x_img2),
            self.la_att_emb(x_img2),
            self.pad_img_att_emb(x_img2)
        ], dim=1)  # [b, 4, seq_len, dim]

        att_tensors3 = torch.stack([
            self.ap_att_emb(x_img3),
            self.pa_att_emb(x_img3),
            self.la_att_emb(x_img3),
            self.pad_img_att_emb(x_img3)
        ], dim=1)  # [b, 4, seq_len, dim]


        # Apply attention embeddings using broadcasting
        # Reshape masks for broadcasting: [b, 4, 1, 1]
        view_masks1 = view_masks1.unsqueeze(-1).unsqueeze(-1)
        view_masks2 = view_masks2.unsqueeze(-1).unsqueeze(-1)
        view_masks3 = view_masks3.unsqueeze(-1).unsqueeze(-1)

        # Weighted sum using masks
        att_img1 = (att_tensors1 * view_masks1).sum(dim=1)  # [b, seq_len, dim]
        att_img2 = (att_tensors2 * view_masks2).sum(dim=1)  # [b, seq_len, dim]
        att_img3 = (att_tensors3 * view_masks3).sum(dim=1)  # [b, seq_len, dim]

        # Step 4: Combine embeddings
        x_img1_final = x_img1 + att_img1 + pos_img1  # [b, n_img1, dim]
        x_img2_final = x_img2 + att_img2 + pos_img2  # [b, n_img2, dim]
        x_img3_final = x_img3 + att_img3 + pos_img3  # [b, n_img3, dim]
        
        # ---

        # !# text; token and positional embeddings
        x_text = self.token_emb(txt)
        x_text += self.txt_att_emb(x_text)
        x_text += self.pos_emb(x_text)

        # FEED x
        x_text_padded = F.pad(x_text, (0, 0, 0, n_img1 - n_txt), "constant", 0)
        x_stacked = torch.stack([x_text_padded, x_img1_final, x_img2_final, x_img3_final], dim=1)

        n_seq = x_text_padded.shape[-1]

        perms = batch['modal_perms']
        batch_indices = torch.arange(b).unsqueeze(1).expand(-1, 4)
        permuted = x_stacked[batch_indices, perms]
        x = permuted.reshape(b, -1, n_seq) # [B, seq_len, dim]

        # dropout layer
        x = self.dropout(x)

        n_condition = n - n_img1

        # performer layers
        layer_pos_emb = self.layer_pos_emb(x)
        x = self.transformer(x, pos_emb=layer_pos_emb, causal=causal, condition_len=n_condition, **kwargs)  # x: [B, seq_len, dim] -> [B, seq_len, dim]
        x = self.norm(x)

        if return_encodings:  # usually False
            return x

        if self.attn_type in ['all_modality_causal_noncuda', 'all_modality_causal_cuda']:
            return self.to_out_combined_txt_img(x)
        return x @ self.token_emb.weight.t()

    # !# Generate Report
    @torch.no_grad()
    @eval_decorator
    def generate_texts(
            self,
            # img1,  # tensor[B, img1_len]
            # img2,  # tensor[B, img2_len]
            # view,
            # modes,
            batch,
            *,
            sos_token_idx=None,
            eos_token_idx=None,
            pad_token_idx=None,
            filter_logits_fn='top_k',
            filter_thres=0.9,
            temperature=1.,
            causal='conditioned_causal'
    ):
        total_len = self.max_seq_len
        txt, img1, modes, view = batch['txt'], batch['img1'], batch['modes'], batch['view_position']
        B, img1_seq_len, device = *img1.shape, img1.device
        _, txt_seq_len = txt.size()

        if 'img2' in batch.keys():
            assert self.max_img_num >= 2
            img2 = batch['img2']
        if 'img3' in batch.keys():
            assert self.max_img_num == 3
            img3 = batch['img3']

        if self.max_img_num == 1:
            assert modes[0][0] == 'img1'
            images = img1
        elif self.max_img_num == 2:
            if modes[0][0] == 'img1':
                images = torch.cat((img1, img2), dim=1)  # -> [B, image_seq_len]
            elif modes[0][0] == 'img2':
                images = torch.cat((img2, img1), dim=1)  # -> [B, image_seq_len]
            else:
                raise ValueError
        elif self.max_img_num == 3:
            if modes[0][0] == 'img1':
                if modes[1][0] == 'img2':
                    images = torch.cat((img1, img2, img3), dim=1)
                else:
                    images = torch.cat((img1, img3, img2), dim=1)
            elif modes[0][0] == 'img2':
                if modes[1][0] == 'img1':
                    images = torch.cat((img2, img1, img3), dim=1)
                else:
                    images = torch.cat((img2, img3, img1), dim=1)
            elif modes[0][0] == 'img3':
                if modes[1][0] == 'img1':
                    images = torch.cat((img3, img1, img2), dim=1)
                else:
                    images = torch.cat((img3, img2, img1), dim=1)

        B, image_seq_len, device = *images.shape, images.device
        out = torch.cat((images, torch.tensor([[sos_token_idx]] * B).to(device)), dim=-1)
        batch['txt'] = out[:, image_seq_len:]

        if filter_logits_fn == 'top_k':
            filter_logits_fn = top_k
        elif filter_logits_fn == 'top_p':
            filter_logits_fn = top_p
        else:
            raise ValueError('filter_logits_fn must be in (top_k, top_p)')

        for cur_len in range(txt_seq_len - 1):
            batch['txt'] = out[:, image_seq_len:]
            logits = self(batch, causal=causal)
            max_neg_value = -torch.finfo(logits.dtype).max
            logits[:, :, self.num_txt_tokens:] = max_neg_value
            logits = logits[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)  # [B, num_text_tokens]
            sample = torch.multinomial(probs, 1)  # [B, 1]
            out = torch.cat((out, sample), dim=-1)
            # break check
            if ((out[:, image_seq_len:] == eos_token_idx).sum(dim=-1) > 0).sum() == B:
                break

        text_seq = out[:, image_seq_len:]

        # postprocess
        indices = [list(row).index(eos_token_idx) if eos_token_idx in row else -1 for row in text_seq]
        for row, idx in enumerate(indices):
            if idx >= 0:
                text_seq[row, idx + 1:] = pad_token_idx

        batch['txt'] = txt
        pad_size = (0, txt_seq_len - text_seq.size(-1))
        gen_texts = F.pad(text_seq, pad_size, 'constant', pad_token_idx)
        return gen_texts

    # !# Generate Certain Image
    @torch.no_grad()
    @eval_decorator
    def generate_image(self,
                       # txt,
                       # img,
                       # view,
                       # modes,
                       batch,
                       *,
                       filter_logits_fn='top_k',
                       filter_thres=0.9,
                       temperature=1.,
                       causal='conditioned_causal',
                       target_gen_view='AP',
                       ):
        txt, img1, modes, view = batch['txt'], batch['img1'], batch['modes'], batch['view_position']

        if 'img2' in batch.keys():
            assert self.max_img_num >= 2 or self.max_img_num == -1
            img2 = batch['img2']
        if 'img3' in batch.keys():
            assert self.max_img_num == 3 or self.max_img_num == -1
            img3 = batch['img3']

        B, n_txt, device = *txt.shape, txt.device

        att_sos_special_tokens = {'AP': 1025, 'PA': 1027, 'LATERAL': 1029, 'LL': 1029, 'PAD': 1024}

        if self.max_img_num == 1:
            out = txt
        elif self.max_img_num == 2:
            if modes[-1][0] == 'img2':
                if modes[0][0] == 'img1':
                    out = torch.cat((batch['img1'], txt), dim=1).to(device)
                else:
                    out = torch.cat((txt, batch['img1']), dim=1).to(device)
            elif modes[-1][0] == 'img1':
                if modes[0][0] == 'img2':
                    out = torch.cat((batch['img2'], txt), dim=1).to(device)
                else:
                    out = torch.cat((txt, batch['img2']), dim=1).to(device)
        elif self.max_img_num == 3:
            mode_to_data = {'img1': batch['img1'], 'img2': batch['img2'], 'img3': batch['img3'], 'txt': batch['txt']}
            conditioned_data = []
            for mode in modes[:-1]:
                conditioned_data.append(mode_to_data[mode[0]])
            out = torch.cat(conditioned_data, dim=1).to(device)
        B, seq_len, device = *out.shape, out.device
        out = torch.cat([out, torch.tensor([[att_sos_special_tokens[i]] for i in view[-1]]).to(device)], dim=-1)

        if filter_logits_fn == 'top_k':
            filter_logits_fn = top_k
        elif filter_logits_fn == 'top_p':
            filter_logits_fn = top_p
        else:
            raise ValueError('filter_logits_fn must be either top_k or top_p')

        for cur_len in range(self.img_len-1):
            batch[modes[-1][0]] = out[:, seq_len:]

            logits = self(batch, causal=causal)
            max_neg_value = -torch.finfo(logits.dtype).max
            logits[:, :, :self.num_txt_tokens] = max_neg_value
            logits = logits[:, -1, :]

            if cur_len != (self.img_len-2):
                logits[:, (self.img_vocab_size + self.num_txt_tokens):] = float('-inf')

            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)  # [B, 1]

            sample -= self.num_txt_tokens

            if cur_len != (self.img_len - 2):
                assert not set(sum(sample.tolist(), [])) & set(range(1024, self.num_img_tokens)), f'{sample}, Special token are sampled in wrong position.'

            out = torch.cat((out, sample), dim=-1)
        image_seq = out[:, seq_len:]

        if modes[-1][0] == 'img1':
            batch[modes[-1][0]] = img1
        elif modes[-1][0] == 'img2':
            batch[modes[-1][0]] = img2
        elif modes[-1][0] == 'img3':
            batch[modes[-1][0]] = img3

        return image_seq
