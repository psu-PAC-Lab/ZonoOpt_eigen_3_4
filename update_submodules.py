import os

os.system('git submodule init')
os.system('git submodule update')

w = os.walk('extern/')
for (dirpath, dirnames, filenames) in w:
    if '.git' in filenames:
        print(f'Updating submodules in {dirpath}')
        os.system(f'cd {dirpath} && git submodule init')
        os.system(f'cd {dirpath} && git submodule update')
