
from numpy import dtype
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
from vit_pytorch import ViT
from vit_pytorch.deepvit import DeepViT
from skimage.io import imread, imshow
from utils import *
from configs import InputParser

def dataloader(dataset,batch_size):
    L = len(dataset[0])
    idx = np.random.randint(L,size=batch_size).astype(int)
    lbl = np.asarray(dataset[1])
    return (dataset[0][idx], lbl[idx])

batch_size=4
args = InputParser()
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_patches = args['n_patches']
n_patches_test = args['n_patches_test']
patch_size = args['patch_size']
IMG_CHANNELS = 1
dataset = args['dataset']
vit_patch=16

trainset = torchvision.datasets.CIFAR10(root='./resources/cifar10', train= True,
                                        download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                        shuffle=True, num_workers=2)


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

X_tst_lbl = data.label_maker(stage='test',
                        n_patches=n_patches,
                        patch_size=patch_size,
                        aug_mode='regular')

XX = dataloader(X_lbl,batch_size)
XX_tst = dataloader(X_tst_lbl,batch_size)

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

v2=DeepViT(
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
img, lbl = XX
# img, lbl = data
img = np.moveaxis(img,-1,1)
# img=torch.unsqueeze(torch.tensor(img[:,:,0],dtype=torch.float),0).unsqueeze(0)
img = torch.tensor(img,dtype=torch.float32)

img_tst, lbl_tst = XX_tst
img_tst = np.moveaxis(img_tst,-1,1)
img_tst = torch.tensor(img_tst,dtype=torch.float32)

# img = torch.randn(1, 3, 256, 256)

target=np.zeros([len(lbl),1])
for i in range(len(lbl)):
    if lbl[i]=='cracks':
        target[i,0]=1

target_tst=np.zeros([len(lbl_tst),1])
for i in range(len(lbl_tst)):
    if lbl_tst[i]=='cracks':
        target_tst[i,0]=1

target=torch.tensor(target)
target1=target.squeeze(-1).long()

target_tst=torch.tensor(target_tst).squeeze(-1).long()

# preds = preds.float()

criterion=torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(v.parameters(), lr=0.001, momentum=0.9)

running_loss = 0.0
# running_loss2 = 0.0
for epoch in range(5):
    optimizer.zero_grad()
    
    preds = F.softmax(v(img),dim=1).float()
    # preds2 = F.softmax(v2(img),dim=1).float()
    loss=criterion(preds,target1)
    # loss2=criterion(preds2,target1)
    loss.backward()
    # loss2.backward()
    optimizer.step()

    running_loss += loss.item()
    # running_loss2 += loss2.item()
    print('[%d] loss: %.3f' % (epoch + 1, running_loss))
    # print('[%d] loss2: %.3f' % (epoch + 1, running_loss2))
    running_loss = 0
    # running_loss2 = 0

preds_tst=v(img_tst)
print(torch.argmax(preds_tst,dim=1))
print(target_tst)
print(criterion(preds_tst,target_tst))
# print(loss)

print("Done!")
