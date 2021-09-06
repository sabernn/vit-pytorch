
import torch
from vit_pytorch import ViT
from skimage.io import imread, imshow
from utils import *
from configs import InputParser

args = InputParser()

n_patches = args['n_patches']
n_patches_test = args['n_patches_test']
patch_size = args['patch_size']
IMG_CHANNELS = 1
dataset = args['dataset']
vit_patch=64

data = DataGenerator(args)
X_train,Y_train,n_patches,_ = data.input_data(args['class'],
                                        'train',
                                        n_patches=n_patches,
                                        patch_size=patch_size,
                                        aug_mode=args['aug_mode'],
                                        plot=True)
X_test,Y_test,_,_ = data.input_data(args['class'],
                                        'test',
                                        n_patches=n_patches_test,
                                        patch_size=patch_size,
                                        aug_mode=args['aug_mode'],
                                        plot=True)

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
img = X_lbl
img = np.moveaxis(img,-1,1)
# img=torch.unsqueeze(torch.tensor(img[:,:,0],dtype=torch.float),0).unsqueeze(0)
img = torch.tensor(img,dtype=torch.float32)

# img = torch.randn(1, 3, 256, 256)

preds = v(img)
print(preds)

print("Done!")
