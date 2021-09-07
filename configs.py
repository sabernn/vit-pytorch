
import argparse


def InputParser():
    parser=argparse.ArgumentParser(description="Vision Transformers for Tomography Data")

    parser.add_argument('--dataset',default='zeiss',help="Defines the tomography image dataset")
    parser.add_argument('--class',default='pores',help="Defines fault type to be identified inside the image")
    parser.add_argument('--n_patches',default=10, type=int, help="Number of training patches for data augmentation.")
    parser.add_argument('--n_patches_test',default=1, type = int, help="Number of test patches for data augmentation.")
    parser.add_argument('--patch_size',default=128, type = int, help="Size of the square patch.")
    parser.add_argument('--rnd_seed',default=1, type = int, help="Random library random seed.")
    parser.add_argument('--tf_rnd_seed',default=1, type = int, help="TensorFlow random seed.")
    parser.add_argument('--epochs',default=10, type = int, help="Number of epochs.")
    parser.add_argument('--batch_size',default=1, type = int, help="Batch size")
    parser.add_argument('--patience',default=3, type = int, help="Number of patience epochs during training.")
    parser.add_argument('--aug_mode',default='regular',type=str)


    args, unknown = parser.parse_known_args()
    args = vars(args)

    return args



