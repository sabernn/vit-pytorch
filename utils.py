from abc import ABC, abstractmethod
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.patches as patches
import os

class DataGeneratorBase(ABC):
    @abstractmethod
    def __init__(self,args):
        '''
        Coming up with a standardized input data format.
        :param args: contains networc parameters including the name of the image dataset (e.g. aps, zeiss, northstar, etc.)
        '''
        self.dataset=args['dataset']


    @abstractmethod
    def input_data(self):
        '''
        :return X: a 4D numpy array containing the image patches and the corresponding color code of each pixel
        :return Y: a 4D numpy array containing the image patches and the corresponding label of each pixel
                 X dimension: (number of samples)x(image height)x(image width)x(number of channels)
                 Y dimension: (number of samples)x(image height)x(image width)x(1)
        '''
        pass


class DataGenerator(DataGeneratorBase):
    def __init__(self, args):
        super().__init__(args)
        self.ROOT_DIR = os.path.abspath("")  # directory of main.py
        self.TRAIN_DIR = os.path.join(self.ROOT_DIR, 'resources', self.dataset, 'train')
        self.TEST_DIR = os.path.join(self.ROOT_DIR, 'resources', self.dataset, 'test')
        self.categories=["original","pores","cracks"]
        self.CAT_TRAIN_DIRS=[""]*len(self.categories)
        self.CAT_TEST_DIRS=[""]*len(self.categories)
        for i,c in enumerate(self.categories):
            self.CAT_TRAIN_DIRS[i]=os.path.join(self.TRAIN_DIR,c)
            self.CAT_TEST_DIRS[i]=os.path.join(self.TRAIN_DIR,c)

        self.IMG_HEIGHT=self.get_image_dims()[0]
        self.IMG_WIDTH=self.get_image_dims()[1]
        self.rnd_seed=args['rnd_seed']
        self.args=args
        


    def show_data(self,stage='train'):
        titles=list(map(lambda x: x.capitalize(), self.categories))
        path_orig=os.path.join(self.TRAIN_DIR,self.categories[0])
        imgcode=next(os.walk(path_orig))[2][0]

        print("$$$$$$$$$$$$")
        print("$$$ NOTE $$$: You have {0} images for {1} stage.".format(len(next(os.walk(path_orig))[2]),stage))
        print("$$$$$$$$$$$$")
        
        for i,c in enumerate(self.categories):
            L=len(self.categories)
            plt.subplot(1,L,i+1)
            plt.title(titles[i])
            plt.xlabel("X")
            plt.ylabel("Y")
            temp_path=os.path.join(self.TRAIN_DIR,c,imgcode)
            img=imread(temp_path)
            imshow(temp_path)
        plt.show()

        pass

    def get_image_dims(self):

        # path_orig=os.path.join(self.TRAIN_DIR,self.categories[0])

        imgcode=next(os.walk(self.CAT_TRAIN_DIRS[0]))[2][0]
        path_img=os.path.join(self.CAT_TRAIN_DIRS[0],imgcode)
        origimg=imread(path_img)

        height=origimg.shape[0]     # number of rows
        width=origimg.shape[1]      # number of columns

        return height,width


    def input_data(self,fault,stage='train',n_patches=100,patch_size=256,aug_mode='random_patches',plot=True):
        super().input_data()
            
        try:
            ind=self.categories.index(fault)
        except:
            raise ValueError("Saber: '{0}' is not among the specified categories in your data model!".format(fault))

        count=0

        IMG_CHANNELS=1  # to be revised later

        if stage=='train':
            random.seed(self.rnd_seed)
            imgcode=next(os.walk(self.CAT_TRAIN_DIRS[ind]))[2][0]
            oimage=imread(os.path.join(self.CAT_TRAIN_DIRS[0],imgcode))
            fimage=imread(os.path.join(self.CAT_TRAIN_DIRS[ind],imgcode))
        elif stage=='test':
            random.seed(self.rnd_seed+1)
            imgcode=next(os.walk(self.CAT_TEST_DIRS[ind]))[2][0]
            oimage=imread(os.path.join(self.CAT_TEST_DIRS[0],imgcode))
            fimage=imread(os.path.join(self.CAT_TEST_DIRS[ind],imgcode))
        else:
            raise ValueError("Saber: 'stage' is not defined!")
        
        ### CONSIDER OTHER METHODS: Otsu's, Gaussian Mixture
        image_thr=thresholder(fimage,254)

        if plot:
            fig,ax=plt.subplots(1)
            ax.imshow(fimage,cmap='gray', vmin=0, vmax=255)

        if aug_mode=='random_patches':
            X=np.zeros((n_patches,patch_size,patch_size,IMG_CHANNELS),dtype=np.uint8)
            Y=np.zeros((n_patches,patch_size,patch_size,1),dtype=np.bool)
            L=[None]*n_patches
            while count<n_patches:
                # plt.figure(2)
                # plt.subplot(countx,county,i+1)
                upperleft_x=random.choice(range(self.IMG_HEIGHT-patch_size))
                upperleft_y=random.choice(range(self.IMG_WIDTH-patch_size))
                # plt.xlabel("Upperleft_x = "+str(upperleft_x))
                # plt.ylabel("Upperleft_y = "+str(upperleft_y))
                # print(upperleft_x)
                # print(upperleft_y)
                img=oimage[upperleft_x:upperleft_x+patch_size,upperleft_y:upperleft_y+patch_size]
                # imshow(img)
                img2=image_thr[upperleft_x:upperleft_x+patch_size,upperleft_y:upperleft_y+patch_size]
                
                # fig,ax=plt.subplots(1)
                # ax.imshow(image)
                
                img=np.expand_dims(img,axis=-1)
                img2=np.expand_dims(img2,axis=-1)

                if np.max(img2)>0:
                    # print(count)
                    X[count]=img
                    Y[count]=img2
                    L[count]=fault
                    count+=1
                    if plot:
                        rect=patches.Rectangle((upperleft_y,upperleft_x),patch_size,patch_size,linewidth=1,
                                        edgecolor='w',facecolor="none")

                        ax.add_patch(rect)


        elif aug_mode=='regular':
            max_patches=int(self.IMG_HEIGHT/patch_size)*int(self.IMG_WIDTH/patch_size)
            X=np.zeros((max_patches,patch_size,patch_size,IMG_CHANNELS),dtype=np.uint8)
            Y=np.zeros((max_patches,patch_size,patch_size,1),dtype=np.bool)
            L=[None]*max_patches
            print(max_patches)
            for i in range(0,self.IMG_HEIGHT-patch_size,patch_size):
                for j in range(0,self.IMG_WIDTH-patch_size,patch_size):
                    upperleft_x=j
                    upperleft_y=i
                    img=oimage[upperleft_x:upperleft_x+patch_size,upperleft_y:upperleft_y+patch_size]
                    img2=image_thr[upperleft_x:upperleft_x+patch_size,upperleft_y:upperleft_y+patch_size]

                    img=np.expand_dims(img,axis=-1)
                    img2=np.expand_dims(img2,axis=-1)

                    if np.max(img2)>0:
                        # print(count)
                        X[count]=img
                        Y[count]=img2
                        L[count]=fault
                        count+=1
                        if plot:
                            rect=patches.Rectangle((upperleft_y,upperleft_x),patch_size,patch_size,linewidth=1,
                                            edgecolor='w',facecolor="none")

                            ax.add_patch(rect)

            X=X[:count]
            Y=Y[:count]
            # L[:count]=fault
            L=L[:count]


        print("Count: "+str(count))

        if plot:
            plt.title("Selected patches for "+stage.capitalize())
            plt.show() 

        return X,Y,count,L

    def label_maker(self,stage='train',n_patches=100,patch_size=256,aug_mode='regular'): # no idea for random mode yet!
        n_classes=len(self.categories)
        n_faults=n_classes-1 # excluding the original image
        n_labels=2**n_faults-1 # all subsets of n_faults minus the null subset
        faults=[]
        for i in range(n_faults):
            imgs,_,_,L=self.input_data(self.categories[i+1],stage=stage,n_patches=n_patches,patch_size=patch_size,aug_mode=aug_mode,plot=False)
            faults.append((imgs,L))
        IMGs=np.concatenate((faults[0][0],faults[1][0]),axis=0)
        LBLs=faults[0][1]+faults[1][1]
        print("stay here...")
        return IMGs



def show_data(dataset='zeiss',stage='train',plot=True):
    # original, pores, cracks
    ROOT_DIR = os.path.abspath("")
    path_stage = os.path.join(ROOT_DIR, 'resources', dataset, stage)
    categories=["original","pores","cracks"]
    titles=["Original", "Pores", "Cracks"]
    category=categories[1]

    path_cat=os.path.join(path_stage,category)
    imgcode=next(os.walk(path_cat))[2][0]

    print("$$$$$$$$$$$$")
    print("$$$ NOTE $$$: You have {0} images for {1} stage.".format(len(next(os.walk(path_cat))[2]),stage))
    print("$$$$$$$$$$$$")

    path_img=os.path.join(path_cat,imgcode)
    origimg=imread(path_img)
    if plot==True:
        for i,c in enumerate(categories):
            plt.subplot(1,3,i+1)
            plt.title(titles[i])
            plt.xlabel("X")
            plt.ylabel("Y")
            temp_path=os.path.join(path_stage,c,imgcode)
            img=imread(temp_path)
            imshow(temp_path)


        plt.show()
    width=origimg.shape[0]
    height=origimg.shape[1]

    return width,height


def thresholder(orig_image,margin):
    img=orig_image.copy()
    img[img<margin]=0
    return img


def cumulative_loss(loss):
    L=len(loss)
    closs=[loss[0]]
    for i in range(L-1):
        closs.append(closs[i]+loss[i+1])

    return closs


def get_flops():
    g = tf.Graph()
    run_meta = tf.compat.v1.RunMetadata()
    with g.as_default():
        A = tf.Variable(tf.random.normal([25,16]))
        B = tf.Variable(tf.random.normal([16,9]))
        C = tf.matmul(A,B)

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.compat.v1.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        if flops is not None:
            print('Flops should be ~',2*25*16*9)
            print('TF stats gives',flops.total_float_ops)


