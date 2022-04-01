from torchvision.datasets import ImageNet
import os
import glob
import shutil
from torchvision.transforms import Resize
from PIL import Image
from multiprocessing import Pool
import argparse


# resize to 256
# during training crop to 224

def resize_image(file, resized_folder, resizer):
    class_dir = os.path.join(resized_folder, '/'.join(file.split('/')[-3:-1]))
    os.makedirs(class_dir, exist_ok=True)

    save_path = os.path.join(resized_folder, '/'.join(file.split('/')[-3:]))
    if not os.path.exists(save_path):

        '''if save_path.split('/')[-1] == 'n02105855_2933.JPEG':
            import IPython
            IPython.embed()'''

        im = Image.open(file)
        resized_im = resizer(im)

        if resized_im.mode in ("RGBA", "P"):
            resized_im = resized_im.convert("RGB")

        resized_im.save(save_path)


def resize_imagenet(root_dir, min_size=256, num_workers=8):
    resizer = Resize(min_size)

    resized_folder = os.path.join(root_dir, 'resized_imagenet')
    os.makedirs(resized_folder, exist_ok=True)

    val_files = glob.glob(os.path.join(root_dir, 'val/*/*.JPEG'))
    train_files = glob.glob(os.path.join(root_dir, 'train/*/*.JPEG'))

    pool = Pool(processes=num_workers)

    if not os.path.isdir(os.path.join(resized_folder, 'val')):
        print('Resizing Validation Data...')
        pool.starmap(resize_image, zip(val_files, [resized_folder]*len(val_files), [resizer]*len(val_files)))

    if not os.path.isdir(os.path.join(resized_folder, 'train')):
        print('Resizing Training Data...')
        pool.starmap(resize_image, zip(train_files, [resized_folder] * len(train_files), [resizer] * len(train_files)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        help='Path to extracted Imagenet data downloaded from official website (https://image-net.org)')
    parser.add_argument('--num_workers', help='Number of workers', type=int, default=8)
    parser.add_argument('--min_size', help='New size of the smaller side of the image', type=int, default=256)

    args = parser.parse_args()
    resize_imagenet(args.data_dir, min_size=args.min_size, num_workers=args.num_workers)

    shutil.copy2(os.path.join(args.data_dir, 'ILSVRC2012_devkit_t12.tar.gz'), os.path.join(args.data_dir, 'resized_imagenet/ILSVRC2012_devkit_t12.tar.gz'))

    print('Finished! You can now add "resized_imagenet" to your root path in the torchvision dataloader for using the resized version')
