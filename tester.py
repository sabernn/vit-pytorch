
from numpy import dtype
import torch
import torch.nn.functional as F
import torch.optim as optim
from vit_pytorch import ViT
from skimage.io import imread, imshow
from utils import *
from configs import InputParser

args = InputParser()
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_patches = args['n_patches']
n_patches_test = args['n_patches_test']
patch_size = args['patch_size']
IMG_CHANNELS = 1
dataset = args['dataset']
vit_patch=16

data = DataGenerator(args)
X_train,Y_train,n_patches,_ = data.input_data(args['class'],
                                        'train',
                                        n_patches=n_patches,
                                        patch_size=patch_size,
                                        aug_mode=args['aug_mode'],
                                        plot=False)
X_test,Y_test,_,_ = data.input_data(args['class'],
                                        'test',
                                        n_patches=n_patches_test,
                                        patch_size=patch_size,
                                        aug_mode=args['aug_mode'],
                                        plot=False)

X_lbl = data.label_maker(stage='train',
                        n_patches=n_patches,
                        patch_size=patch_size,
                        aug_mode='regular')


v=ViT(
    image_size = patch_size,
    patch_size = vit_patch,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    channels = 1,
    dropout = 0.1,
    emb_dropout = 0.1
)

# img=imread('zeiss.tiff')
img, lbl = X_lbl
img = np.moveaxis(img,-1,1)
# img=torch.unsqueeze(torch.tensor(img[:,:,0],dtype=torch.float),0).unsqueeze(0)
img = torch.tensor(img,dtype=torch.float32)

# img = torch.randn(1, 3, 256, 256)

target=np.zeros([len(lbl),1])
for i in range(len(lbl)):
    if lbl[i]=='cracks':
        target[i,0]=1

target=torch.tensor(target)
target1=target.squeeze(-1).long()



# preds = preds.float()

criterion=torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(v.parameters(), lr=0.001, momentum=0.9)

running_loss = 0.0
for epoch in range(20):
    optimizer.zero_grad()
    
    preds = F.softmax(v(img),dim=1).float()
    loss=criterion(preds,target1)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    print('[%d] loss: %.3f' % (epoch + 1, running_loss))
    running_loss = 0


# print(loss)

print("Done!")
