import torch
from torch import nn
from math import ceil

class BlockwiseParallelMoE(nn.Module):
    def __init__(self, dim, dim_inner, num_experts, capacity_factor, min_capacity):
        super(BlockwiseParallelMoE, self).__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.dim_inner = dim_inner
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity

        self._router = nn.Sequential(
            nn.Linear(self.dim, num_experts, bias=False).float(),
            nn.Softmax(dim=-1)        
        )
        self._experts = [
            nn.Sequential(
                nn.Linear(self.dim, self.dim_inner),
                nn.GELU(),
                nn.Linear(self.dim_inner, self.dim)
            )
            for _ in range(num_experts)
        ]

    def _capacity(self, num_tokens):
        return max(self.min_capacity,
                   ceil(num_tokens * self.capacity_factor / self.num_experts))

    #TODO: can I torch.jit.compile this? 
    def _expert_choice_routing(self, x, num_tokens):
        k = self._capacity(num_tokens)
        S = self._router(x)
        G, topk_idx = torch.topk(S.T, k, dim=1)
        P = nn.functional.one_hot(topk_idx, num_tokens).to(dtype=torch.float32) # e x k x n
        X_in = torch.matmul(P, x)
        return G, P, X_in

    def _calculate_expert_output(self, G, P, X_in):
        # TODO: Rewrite X_e calculation with einsum
        X_e = torch.stack([self._experts[i](X_in[i]) for i in range(len(self._experts))], dim=0)
        return torch.einsum('ijl,ij,ijd->ld', P, G, X_e)

    def forward(self, x):
        # Expert's choice routing
        # Reshape input to combine batch_size and seq_len into single dimension n = num_tokens
        batch_size, seq_len, embed_dim = x.shape
        x = x.view(-1, x.size(-1))
        G, P, X_in = self._expert_choice_routing(x, x.shape[0])
        X_out = self._calculate_expert_output(G, P, X_in)
        return X_out.reshape(batch_size, seq_len, embed_dim)
