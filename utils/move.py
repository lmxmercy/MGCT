# import os, shutil

# file_list = []

# # 搜索函数
# def search_file(root, target):
#     for file in os.listdir(root):
#         path = root
#         try:
#             path = path + os.sep + file
#             if os.path.isdir(path):
#                 search_file(path, target)
#             else:
#                 if file.split('.')[-1] == target:
#                     file_list.append(path)
#         except PermissionError as e:
#             print(e)
#     return file_list


# # 批量移动函数
# def move_file(file_list, dest):
#     for file in file_list:
#         try:
#             shutil.move(file, dest)
#         except shutil.Error as e:
#             print(e)


# # 写入目标参数root,
# def main():
#     root = "/data_new/lmx/Dataset/TCGA-DATASET/DATA_DIRECTORY/tcga_gbmlgg"
#     target = "svs"
#     dest_dir = "/data_new/lmx/Dataset/TCGA-DATASET/DATA_DIRECTORY/tcga_gbmlgg"
#     result = search_file(root, target)
#     print(result)
#     move_file(result, dest_dir)


# if __name__ == '__main__':
#     main()


# import os

# # 文件夹路径
# img_path = '/data_new/lmx/Dataset/TCGA-DATASET/DATA_DIRECTORY/tcga_gbmlgg'
# # txt 保存路径
# save_txt_path = '/data/lmx/MCAT/utils/names.txt'

# # 读取文件夹中的所有文件

# imgs = os.listdir(img_path)

# # 图片名列表
# names = []

# # 过滤：只保留png结尾的图片
# for img in imgs:
#     if img.endswith(".svs"):
#         names.append(img)

# txt = open(save_txt_path,'w')

# for name in names:
#     name = name[:-4]    # 去掉后缀名.png

#     txt.write(name + '\n')  # 逐行写入图片名，'\n'表示换行

# txt.close()

def file_same():
    str1 = []
    file1 = open("/data/lmx/MCAT/utils/names.txt","r",encoding="utf-8")
    for line in file1.readlines():    #读取第一个文件
        str1.append(line.replace("\n",""))

    str2 = []
    file2 = open("/data/lmx/MCAT/utils/ucec.txt", "r", encoding="utf-8")
    for line in file2.readlines():   #读取第二个文件
        str2.append(line.replace("\n", ""))

    str_dump = []
    a=0
    for line in str1:
        if line in str2:
            str_dump.append(line)    #将两个文件重复的内容取出来
            print(line)    #将重复的内容输出
            a=a+1
            print(",,,,,,,,,,,,,,,,,,,,,,,,,")

    str_all = set(str1 + str2)      #将两个文件放到集合里，过滤掉重复内容
    #print(a)

    for i in str_dump:
        if i in str_all:
            str_all.remove(i)		#去掉两个文件中重复的内容

    for str in str_all:             #去重后的结果写入文件
        #print(str)
        with open("/data/lmx/MCAT/utils/ucec_download.txt","a+",encoding="utf-8") as f:
            f.write(str + "\n")

if __name__=="__main__":
    file_same()

