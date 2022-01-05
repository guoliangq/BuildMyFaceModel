import os
import shutil

source_path = "D:\\recordsystem\\facephoto1"
target_path = "D:\\data_test\\three_model"

def copyAndRenameImage(img_name):
    for i in range(0,1000) :
        # 复制图片至目标文件夹下
        shutil.copy(source_path + "\\" + img_name, target_path)
        os.rename(target_path+"\\"+img_name,target_path+"\\"+str(i)+".jpg")

if __name__ == "__main__":
    copyAndRenameImage("20175119.jpg")
