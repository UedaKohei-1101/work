import os
import glob

path = "./Chessboards"
files = glob.glob(path + '/*')

for i, f in enumerate(files):
    os.rename(f, os.path.join(path, '{0:03d}'.format(i) + '.png'))
    