import os
import os.path as osp
test_root = "D:\\data\\CelebA_Spoof\\Data\\test"

'''
该函数的主要功能就是根据CelebA_Spoof测试数据集生成对应的标签文件
文件路径 类型的形式
3613/spoof/511091.png 1
其中类型用0，1表示，0 表示是假脸，1表示是真脸
'''
def generate_test_pair():
    test_pair_file = open("celebA_Spoof_test_pair.txt")
    paths = os.listdir(test_root)
    for path in paths:
        list = os.listdir(osp.join(test_root,path))
        for p in list:
            if p == 'spoof':
                files_in_p = os.listdir(osp.join(osp.join(test_root,path),p))
                for file in files_in_p:
                    #如果以png结尾表示是图片
                    if file.__contains__("png"):
                        print(f"{path}/{p}/{file} 1\n")
                        test_pair_file.write(f"{path}/{p}/{file} 1\n")
            elif p == 'live':
                files_in_p = os.listdir(osp.join(osp.join(test_root,path),p))
                for file in files_in_p:
                    if file.__contains__("png"):
                        print(f"{path}/{p}/{file} 1\n")
                        test_pair_file.write(f"{path}/{p}/{file} 1\n")
            else:
                print(f"something different {p}.\n")
    test_pair_file.close()

'''
if __name__ == '__main__':
    generate_test_pair() 
'''