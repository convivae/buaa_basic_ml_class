import os
import shutil
from random import sample

if __name__ == '__main__':
    # l = range(10)
    # print(type(l))
    # sample_list = sample(l, 3)
    # sample_list.sort()
    # for i in sample_list:
    #     print(i)
    # print(sample_list)

    print(os.path.exists('./depthes'))
    shutil.rmtree(path='./depthes')
    print(os.path.exists('./depthes'))
