
import torch
from torch.functional import Tensor
import torch.nn.functional as F
from skimage.io import imread, imshow
import matplotlib.pyplot as plt


batch_size=1
seq_length=2000
num_features=2000

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