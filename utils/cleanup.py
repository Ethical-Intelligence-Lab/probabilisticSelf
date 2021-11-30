from os import listdir
from os.path import isfile, join
from os import walk

def get_files(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

def get_all(path):
    return [x[0] for x in walk(path)]

if __name__ == '__main__':
    filenames = get_all("/Users/kaan/Documents/GitHub/probabilisticSelf2/saved_models")
    print(filenames)