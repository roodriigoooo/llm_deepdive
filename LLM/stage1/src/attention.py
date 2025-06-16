import torch

class SelfAttentionBasic(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_k = torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_v = torch.nn.Parameter(torch.randn(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_k
        queries = x @ self.W_q
        values = x @ self.W_v
        scores = queries @ keys.T
        attn_weights = torch.nn.functional.softmax(scores/keys.shape[-1]**0.5, dim=1)
        context = attn_weights @ values
        return context

class SelfAttentionBasicv2(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)
        scores = queries @ keys.T
        attn_weights = torch.nn.functional.softmax(scores/keys.shape[-1]**0.5, dim=1)
        context = attn_weights @ values
        return context

class CausalAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_rate=0.1, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # linear projections for queries, keys and values
        self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        # dropout layer for attention weights
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        # causal mask to prevent attending to future tokens
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, inputs: torch.Tensor):
        b, seq_len, d_in = inputs.shape # shape: (batch_size, seq_len, d_in)

        keys = self.W_k(inputs) # shape: (batch_size, seq_len, d_in)
        queries = self.W_q(inputs)
        values = self.W_v(inputs)

        attn_scores = queries @ keys.transpose(-2, -1) # shape: (batch_size, seq_len, seq_len)
        mask = self.mask[:seq_len, :seq_len].bool() #slice mask to match sequence length
        attn_scores.masked_fill_(mask, -torch.inf)

        attn_weights = self.dropout(torch.nn.functional.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1))
        context = attn_weights @ values
        return context

class MultiHeadAttentionSequential(torch.nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout_rate=0.1, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout_rate, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout_rate=0.1, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0) # d_out must be divisible by num_heads
        self.d_out = d_out
        self.num_heads = num_heads
        self.d_head = d_out // num_heads # dimension of each head

        #separate linear projections for queries, keys and values
        self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        # dropout on attention probs
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # causal mask, ones above the diagonal
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer('mask', mask)
        # projection of concatenated heads back to model dim
        self.out_proj = torch.nn.Linear(d_out, d_out) # a linear layer to combine all head outputs
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        """
        x: (batch_size, seq_len/num_tokens, d_in)
        returns: (batch_size, seq_len, d_out)
        """
        b, num_tokens, d_in = x.shape
        keys = self.W_k(x) # shape: (batch_size, seq_len/num_tokens, d_in)
        queries = self.W_q(x)
        values = self.W_v(x)

        # split the keys, queries and values into num_heads heads.
        # (B, T, H, Dh) -> (B, H, T, Dh)
        def split_heads(tensor):
            """
            split Q,K,V into heads and permute to (batch_size, num_heads, seq_len, d_in)
            """
            return (
                tensor
                .reshape(b, num_tokens, self.num_heads, self.d_head)
                .permute(0, 2, 1, 3)
            )

        queries, keys, values = split_heads(queries), split_heads(keys), split_heads(values)


        #the transpose transposes from shape (batch_size, num_heads, num_tokens, d_head) to (batch_size, num_tokens, num_heads, d_head)
        attn_scores = queries @ keys.transpose(-2, -1) #--> (batch_size, num_heads, num_tokens, d_head)
        mask = self.mask[:num_tokens, :num_tokens] #slice mask to match sequence length
        attn_scores_masked = attn_scores.masked_fill(mask, -torch.inf)

        #softmax and dropout
        attn_weights = self.dropout(torch.nn.functional.softmax(attn_scores_masked * self.scale, dim=-1))
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ values # shape: (batch_size, num_heads, num_tokens, d_head)
        context = context.permute(0, 2, 1, 3).reshape(b, num_tokens, self.d_out)
        context = self.out_proj(context)
        return context