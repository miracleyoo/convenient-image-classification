# coding: utf-8
import pickle
import shutil
import os
from utils import load_class_name
from config import Config
from pathlib import Path
# import paramiko
# from scp import SCPClient

opt = Config()

# #创建ssh访问
# port = 22
# hostname = '115.156.207.244'
# username = 'dian'
# password = 'DianSince2002'
# local_path = './source/bad_case/%s_bad_case.pkl'%opt.DATASET_PATH
# remote_path = '~/miracle/auto-wash/source/bad_case/%s_bad_case.pkl'%opt.DATASET_PATH

# def createSSHClient(hostname, port, username, password):
#     client = paramiko.SSHClient()
#     client.load_system_host_keys()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     client.connect(hostname, port, username, password)
#     return client

# ssh = createSSHClient(hostname, port, username, password)
# scp = SCPClient(ssh.get_transport())
# scp.get(local_path=local_path, remote_path=remote_path)

# bad_case_names = pickle.load(open(local_path, 'rb'))

results = pickle.load(open('./source/%s_results.pkl'% str(opt.DATASET_PATH), 'rb'))
class_names = load_class_name()

new_path = Path("./Datasets") / opt.DATASET_PATH / 'Classified'
if not os.path.exists(new_path): os.mkdir(new_path)
for class_name in class_names:
    print(new_path/class_name)
    if not (new_path/class_name).exists(): (new_path/class_name).mkdir()

for result in results:
    temp_path = new_path/result[0]
    print(temp_path, result[1])
    shutil.copy(result[1], temp_path)


# dataset_path = './source/bad_case_images/%s'%opt.DATASET_PATH
# if not os.path.exists(dataset_path): os.mkdir(dataset_path)
# folders = list(set([x.split('/')[-2] for x in bad_case_names]))
# for folder in folders:
#     temp_path = './source/bad_case_images/'+ opt.DATASET_PATH + '/' + folder
#     if not os.path.exists(temp_path): os.mkdir(temp_path)

# for path in bad_case_names:
#     new_path = './source/bad_case_images/' + opt.DATASET_PATH + '/' + '/'.join(path.split('/')[-2:])
#     shutil.copyfile(path, new_path)
#     # os.remove(bad_case_names)
#     print("%s has been removed."%path)
# print("==> All bad cases have been removed.")