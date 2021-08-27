
import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F
from skimage.io import imread, imshow
import matplotlib.pyplot as plt


class AttentionHead(nn.Module):
    def __init__(self,dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self,query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim = -1)
            )

def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    pos=torch.arange(seq_len, dtype=torch.float, device=device).reshape(1,-1,1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1,1,-1)
    phase = pos / 1e4 ** (dim // dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


batch_size=1
seq_length=2000
num_features=2000

Attention=AttentionHead(seq_length,seq_length,seq_length)

img=imread('zeiss.tiff')


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tensor:
    temp=query.bmm(key.transpose(1,2))
    scale=query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


query=torch.rand([batch_size,seq_length,num_features],dtype=torch.float)
key=torch.randn([batch_size,seq_length,num_features],dtype=torch.float)
value=torch.ones([batch_size,seq_length,num_features],dtype=torch.float)

img_tensor=torch.unsqueeze(torch.tensor(img[:,:,0],dtype=torch.float),0)
# torch.unsqueeze(img_tensor)

atten=scaled_dot_product_attention(query,key,img_tensor)
print(atten.shape)

fig = plt.figure()
fig.add_subplot(1, 2, 1)
imshow(img)
fig.add_subplot(1, 2, 2)
imshow(atten[0].numpy())
plt.show()

print("done!")