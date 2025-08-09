import os
from glob import glob

in_dir = '../encode'
out_dir = '.'
modality = 'T1'
in_theta = '../encode/SAMPLE_T1_theta.txt'

imgs = os.path.join(in_dir, f'*{modality.upper()}*_ori.nii.gz')
imgs = sorted(glob(imgs))
num_imgs = len(imgs)

for img_id, img in enumerate(imgs):
    prefix = os.path.basename(img)
    prefix = prefix.replace('_ori.nii.gz', '')
    print(f'Processing: {prefix}...')
    
    cmd = f'python ../code/decode_3d.py ' + \
        f'--in-beta {os.path.join(in_dir, prefix)}_beta_axial.nii.gz '+ \
        f'{os.path.join(in_dir, prefix)}_beta_coronal.nii.gz ' + \
        f'{os.path.join(in_dir, prefix)}_beta_sagittal.nii.gz ' + \
        f'--in-theta {in_theta} ' + \
        f'--out-dir {out_dir} ' + \
        f'--prefix {prefix} ' + \
        f'--pretrained-model ../models/harmonization.pt ' + \
        f'--gpu 1 ' + \
        f'--beta-dim 4 ' + \
        f'--theta-dim 2'
    os.system(cmd)

