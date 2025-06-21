import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from dataclasses import dataclass
import tiktoken

@dataclass
class GPTConfig124:
    def __init__(self, vocab_size, context_length, emb_dim, n_heads, n_layers, dropout, qkv_bias):
        self.vocab_size: int = vocab_size
        self.context_length: int = context_length
        self.emb_dim: int = emb_dim
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers
        self.dropout: float = dropout
        self.qkv_bias: bool = qkv_bias

class GELU(nn.Module):
    """
    Use the tanh-based approximation for efficiency.
    """
    def __init__(self):
        super().__init__()
        # we precompute constants to avoid repeated tensor allocations
        self._sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi))
        self._coef = 0.044715


    def forward(self, x):
        return 0.5 * x * (
                1.0 + torch.tanh(self._sqrt_2_over_pi * (x + self._coef * x.pow(3))
                                 )
        )

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.emb_dim
        d_ff = 4 * d_model
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.layers(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-6 #small constant to prevent division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) #gamma and beta, our trainable parameters

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig124):
        super().__init__()
        # 1) multi-head self-attention sublayer implemented before
        self.att = MultiHeadAttention(
            d_in = cfg.emb_dim,
            d_out = cfg.emb_dim,
            context_length = cfg.context_length,
            num_heads = cfg.n_heads,
            dropout_rate = cfg.dropout,
            qkv_bias = cfg.qkv_bias)
        # 2) position-wise feed-forward layer
        self.ff = FeedForward(cfg)
        # 3) pre-norm layernorms
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        # 4) dropout on the residual connection
        self.drop_shortcut = nn.Dropout(cfg.dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x) # normalize before attention (pre-norm)
        x = self.att(x) # (batch_size, seq_len, d)
        x = self.drop_shortcut(x)
        x = x + shortcut # residual

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x) # two-layer MLP with activation & dropout
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig124, tie_weights: bool=True):
        super().__init__()
        # token embedding: from vocab indices to vectors
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        # learnable per-position vectors
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.dropout)
        # a stack of transformer blocs
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        # final layer norm before head
        self.final_form = LayerNorm(cfg.emb_dim)
        # ourput head (maps back to vocab logits)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias = False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # first, lookup token indices -> (batch_size, seq_len, D)
        tok_embeds = self.tok_emb(in_idx)
        # create position indices 0..T-1 -> (T,) then embed -> (T, D)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        # add dropout and run through transformer stack
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        # final form and linear head --> logits (B, T, V)
        x = self.final_form(x)
        logits = self.out_head(x)
        return logits

def generate_text(model, idx, max_new_tokens, context_size):
    """
    :param model: our GPT model, nn.Module
    :param idx: a (batch, num_tokens) array of indices in the current context
    :param max_new_tokens: user-specified maximum number of new tokens
    :param context_size: context size. We cap the current context if it exceeds this size.
    :return: (batch_size, num_tokens + max_new_tokens)
    """
    device = next(model.parameters()).device
    generated = idx.to(device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # only use the last n tokens as context if context size is exceeded.
            context = generated[:, -context_size:]
            logits = model(context)
            last_logits = logits[:, -1, :]
             # focus only on last time step.
            # (batch_size, num_tokens, vocab_size) -> (batch_size, vocab_size)
            probas = torch.softmax(last_logits, dim=-1)
             # (batch_size, vocab_size)
            next_token = torch.argmax(probas, dim=-1, keepdim=True)
            # (batch, 1)
            generated = torch.cat([generated, next_token], dim=1)
            # append to running sequence
    return generated


text = """A man told me"""

tokenizer = tiktoken.get_encoding('gpt2')
encoded = tokenizer.encode(text)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor:", encoded_tensor.shape)
torch.manual_seed(123)

cfg = GPTConfig124(vocab_size=50257, context_length=512, emb_dim=768,
                   n_heads=12, n_layers=12, dropout=0.1, qkv_bias=False)
model = GPTModel(cfg)

model.eval()
out = generate_text(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=20,
    context_size=cfg.context_length,
)
print("output:", out)
print("output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze().tolist())
print(decoded_text)
