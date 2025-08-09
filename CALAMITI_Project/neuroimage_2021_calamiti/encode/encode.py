import os
from glob import glob

in_dir = '../sample_dataset/volumes'
modality_names = ['T1', 'T2']

for modality_name in modality_names:
    imgs = os.path.join(in_dir, f'*{modality_name.upper()}*norm.nii.gz')
    imgs = sorted(glob(imgs))
    num_imgs = len(imgs)
    for img_id, img in enumerate(imgs):
        prefix = os.path.basename(img)
        prefix = prefix.replace('_norm.nii.gz', '')
        print(f'{str(img_id+1)}/{str(num_imgs)} Processing: {prefix}')
        cmd = 'python ../code/encode_3d.py ' + \
                f'--in-img {img} ' + \
                f'--out-dir . ' + \
                f'--pretrained-model ../models/harmonization.pt ' + \
                f'--prefix {prefix} ' + \
                f'--avg-theta ' + \
                f'--norm 1000 ' + \
                f'--gpu 1 ' + \
                f'--beta-dim 4 ' + \
                f'--theta-dim 2'
        os.system(cmd)
