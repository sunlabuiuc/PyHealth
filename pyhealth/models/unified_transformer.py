# Adapted from https://github.com/MickEnev/UniXGen/blob/adb91b56456b2858dfa56a5ccba8a981ff8726d5/transformer_pytorch/FAVOR_unified.py

import math
from functools import partial

from torch.cuda.amp import autocast

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

from transformer_pytorch.model_utils import *

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)
    return data_dash.type_as(data)


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=0.001, normalize_data=True, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


def all_modality_causal_linear_attn_noncuda(q, k, v, condition_len, chunk_size=128, eps=1e-6):  # q, k: [B, global_head, seq_len, nb_features]  v: [B, global_head, seq_len, dim_head]
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []
    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim=-2), (q, k, v))):  # q,k:[B, global_head, seq_len/chunk_size, nb_features]  v:[B, global_head, seq_len/chunk_size, dim_head]
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)  # [B, global_head, seq_len/chunk_size, nb_features]
        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)  # -> [B, global_head, seq_len/chunk_size]

        context = torch.einsum('...nd,...ne->...nde', k, v)  # -> [B, global_head, seq_len/chunk_size, nb_features ,dim_head] outer product.
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)  # [B, global_head, seq_len/chunk_size, nb_features ,dim_head]

        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)  # -> [B, global_head, seq_len/chunk_size, dim_head]

        last_k_cumsum = k_cumsum[:, :, -1:]  # [B, global_head, 1, nb_features]
        last_context_cumsum = context_cumsum[:, :, -1:]  # [B, global_head, 1, nb_features ,dim_head]
        outs.append(out)

    return torch.cat(outs, dim=-2)  # -> [B, global_head, seq_len, dim_head]
    
def all_modality_causal_linear_attn_cuda(q, k, v, condition_len, eps=1e-6):
    try:
        from fast_transformers.causal_product import CausalDotProduct
    except:
        raise ImportError("missing soft dependency fast_transformers, you can install with `pip install pytorch-fast-transformers`")
    
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled=False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out


class FastAttention(nn.Module):
    def __init__(
            self,
            dim_head,
            nb_features=None,
            ortho_scaling=0,
            causal=False,
            attn_type="conditioned_noncuda",
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            no_projection=False
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_head * math.log(dim_head)))

        self.dim_head = dim_head
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features,
                                         nb_columns=dim_head, scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection

        self.causal = causal
        if self.causal == 'causal_linear_attn_cuda':
            self.causal_linear_fn = partial(all_modality_causal_linear_attn_noncuda)
        else:
            # In case of running on CUDA device, use Apex optimization to speed-up
            # parallel CausalDotProduct,
            # NOTE: this requires the fast-transformers library.
            self.causal_linear_fn = partial(all_modality_causal_linear_attn_cuda)

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v, condition_len=0):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn,
                                    projection_matrix=self.projection_matrix, device=device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = self.causal_linear_fn
        out = attn_fn(q, k, v, condition_len=condition_len)

        out = out.to(device)
        return out  # [B, global_head, seq_len, dim_head]


class FAVORAttention(nn.Module):
    def __init__(
            self,
            dim,
            causal=False,
            attn_type="conditioned_noncuda",
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            heads=8,
            local_heads=0,
            nb_features=None,
            dropout=0.3,
            no_projection=False,
            qkv_bias=False,
            attn_out_bias=True
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = (dim // heads)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.dim_head = dim_head

        # Fast Attention
        self.fast_attention = FastAttention(dim_head, nb_features,
                                            causal=causal, attn_type=attn_type,
                                            generalized_attention=generalized_attention,
                                            kernel_fn=kernel_fn,
                                            no_projection=no_projection)

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_output = nn.Linear(dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb=None, context=None, mask=None, context_mask=None, condition_len=0, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        context = default(context, x)
        context_mask = default(context_mask, mask)

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))  # => q, k, v: [B, head, seq_len, dim_head]
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]  # [B, 1, seq_len, 1]
                v.masked_fill(~global_mask, 0.)

            if exists(pos_emb):
                q, k = apply_rotary_pos_emb(q, k, pos_emb)

        out = self.fast_attention(q, k, v, condition_len=condition_len)  # [B, global_head, seq_len, dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')  # -> [B, seq_len, inner_dim]

        out = self.to_output(out)  # -> [B, seq_len, dim]
        return self.dropout(out)  # -> [B, seq_len, dim]


class ProjectionUpdater(nn.Module):
    def __init__(self, instance, feature_redraw_interval):
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projections_(self):
        self.feature_redraw_interval = None

    def redraw_projections(self):
        model = self.instance

        if not self.training:
            return

        if exists(
                self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(model)

            fast_attentions = find_modules(model, FastAttention)  # list
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplemented
