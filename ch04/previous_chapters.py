import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, \
            'd_out must be divisible by num_heads'

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        self.W_q = nn.Linear(d_in, d_out,bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out,bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out,bias=qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length),diagonal=1))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)


    def forward(self,x):
        b, nseq, d_in = x.shape
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        #now split along n_heads
        query = query.view(b,nseq,self.n_heads,self.head_dim)
        key = key.view(b,nseq,self.n_heads,self.head_dim)
        value = value.view(b,nseq,self.n_heads,self.head_dim)

        #move n_head dimension in front of n_seq dim
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        #next is dot product attention scores, matmul over the head_dim dimension
        # (b,n_head,n_seq,d_head) x (b,n_head,d_head,n_seq)
        # output is (b,n_head,n_seq,n_seq')
        omega = query @ key.transpose(-2,-1)

        mask = self.mask.bool()[:nseq, :nseq]
        omega = omega.masked_fill(mask, -torch.inf)
        alpha = torch.softmax(omega/self.head_dim**0.5,dim = -1)
        alpha = self.dropout(alpha)
        # shapes are 
        # alpha ~ (b,n_head,n_seq,n_seq')
        # value ~ (b,n_head,n_seq,d_head) 
        #context_vec ~ (b,n_head,d_seq,d_head)
        context_vec = alpha @ value
        #now put n_head next to d_head to roll back up
        context_vec = context_vec.transpose(1,2)
        context_vec = context_vec.contiguous().view(b,nseq,self.d_out)

        #now it is (b,n_seq,d_out), same shape as x except d_in -> d_out
        context_vec = self.out_proj(context_vec)
        return context_vec