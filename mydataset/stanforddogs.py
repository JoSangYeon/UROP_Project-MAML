"""
http://vision.stanford.edu/aditya86/ImageNetDogs/

Number of categories: 120
Number of images: 20,580
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import pandas as pd
import random


class StanFordDogs(Dataset):
    def __init__(self, root, mode, batch_size, n_way, k_shot_spt, k_shot_qry, resize, start_idx=0):
        super(StanFordDogs, self).__init__()
        
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot_spt = k_shot_spt
        self.k_shot_qry = k_shot_qry
        self.spt_size = self.n_way * self.k_shot_spt # N-way K-shot Support Set은 N*K개의 Samples를 가짐
        self.qry_size = self.n_way * self.k_shot_qry # N-way K-shot Query Set은 N*K개의 Samples를 가짐
        self.resize = resize
        self.start_idx = start_idx # 임의의 라벨 번호 0부터 시작 권장(Class가 100개라면, 0~99까지)
        
        print("StanFord Dogs Dataset({}) :\n\tBatch_size : {}\n\tSupport sets : {}-way {}-shot\n\tQuery sets : {}-way {}-shot\n\tResizing Image : {}x{}".format(
        mode, batch_size, n_way, k_shot_spt, n_way, k_shot_qry, resize, resize))
        
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                             transforms.Resize((self.resize, self.resize)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        self.path = os.path.join(root, 'images') # ex) MiniImageNet/images
        csv_data = self.load_CSV(os.path.join(root, mode+'.csv')) # ex) MiniImageNet/train.csv
        
        self.data, self.img2label, self.cls_num = self.get_img2label(csv_data)
        
        self.spt_x_batch, self.qry_x_batch = self.create_batch(self.batch_size)
        
        
    def load_CSV(self, path):
        csv_file = pd.read_csv(path) # columns = [file_path, label]
        
        dict_labels = {}
        for i in range(len(csv_file)):
            file_name, label = csv_file.iloc[i, :2] # file_path, label
            
            if label in dict_labels.keys():
                dict_labels[label].append(file_name)
            else:
                dict_labels[label] = [file_name]
                
        """
        ex)
        dict_labels = {n0xxxxxx1 : [n0xxxxxx10000.jpg, n0xxxxxx10001.jpg ...],
                       n0xxxxxx2 : [n0xxxxxx20000.jpg, n0xxxxxx20001.jpg ...], ....}
        """
        return dict_labels
    
    def get_img2label(self, csv_data):
        data = [] # 이미지의 경로를 담은 데이터
        img2label = {} # 이미지를 입력으로 label값을 받도록 하는 dict
        for i, (k, v) in enumerate(csv_data.items()):
            data.append(v) # ex) [[n0xxxxxx10000.jpg, n0xxxxxx10001.jpg, ...], [n0xxxxxx20000.jpg, n0xxxxxx20001.jpg, ...], [] [] ....]
            img2label[k] = i + self.start_idx # ex) {n0xxxxxx1 : 0, n0xxxxxx2 : 1, n0xxxxxx3 : 2 ,,,,}
            
        cls_num = len(data)
        return data, img2label, cls_num
    
    def create_batch(self, batch_size):
        """
        create batch for meta-learning.
        즉, batch_size만큼 Random하게 Dataset을 만듦
        ex) batch_size : 1000이면, 1000번 랜덤하게 [Support set n-way k-shot, Query set n-way k-shot]을 만듦
        :return:
        """
        support_x_batch = []
        query_x_batch = []
        
        for b in range(batch_size):
            # 1. select n_way classes randomly(랜덤하게 n개의 class를 뽑음)
            selected_cls = np.random.choice(self.cls_num, self.n_way, replace=False) # No dulicate
            np.random.shuffle(selected_cls)
            
            support_x = []
            query_x = []
            
            for cls in selected_cls:
                # 2. select k_shot_spt, k_shot_qry for each class (Support, Query에 쓰일 Img를 뽑음)
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot_spt + self.k_shot_qry, replace=False)
                np.random.shuffle(selected_imgs_idx)
                
                spt_imgs_idx = np.array(selected_imgs_idx[:self.k_shot_spt]) # ex) [13, 25, 35]
                qry_imgs_idx = np.array(selected_imgs_idx[self.k_shot_spt:]) # ex) [17, 4]
                
                spt_imgs = np.array(self.data[cls])[spt_imgs_idx].tolist() # ex) [n0xxxxxx20013.jpg, n0xxxxxx20025.jpg, n0xxxxxx20035.jpg]
                qry_imgs = np.array(self.data[cls])[qry_imgs_idx].tolist() # ex) [n0xxxxxx20017.jpg, n0xxxxxx20004.jpg]
                
                support_x.append(spt_imgs)
                query_x.append(qry_imgs)
                
            random.shuffle(support_x) # (n-way, k_shot_spt)
            random.shuffle(query_x)   # (n-way, k_shot_qry)
            
            support_x_batch.append(support_x) # (batch, n-way, k_shot_spt)
            query_x_batch.append(query_x) # (batch, n-way, k_shot_qry)
            
        return support_x_batch, query_x_batch
    
    def __getitem__(self, idx):
        spt_x = torch.FloatTensor(self.spt_size, 3, self.resize, self.resize) # (N*K_spt, ch, h, w)
        spt_y = np.zeros((self.spt_size), dtype=int) # (N*K_spt)
        
        qry_x = torch.FloatTensor(self.qry_size, 3, self.resize, self.resize) # (N*K_qry, ch, h, w)
        qry_y = np.zeros((self.qry_size), dtype=int) # (N*K_qry)
        
        # self.spt_x_batch[idx] : 해당 idx번째에 학습할 Meta-Train Dataset(Support) -> (n-way, k_shot)
        # sublist : (k-shot,)
        flatten_spt_x = [os.path.join(self.path, item) for sublist in self.spt_x_batch[idx] for item in sublist]
        flatten_spt_y = np.array([self.img2label[item[:9]] for sublist in self.spt_x_batch[idx] for item in sublist]).astype(np.int32)
        
        flatten_qry_x = [os.path.join(self.path, item) for sublist in self.qry_x_batch[idx] for item in sublist]
        flatten_qry_y = np.array([self.img2label[item[:9]] for sublist in self.qry_x_batch[idx] for item in sublist]).astype(np.int32)
        
        unique = np.unique(flatten_spt_y)
        random.shuffle(unique) # np.unique()는 Sorted됨
        for idx, tag in enumerate(unique):
            spt_y[flatten_spt_y == tag] = idx
            qry_y[flatten_qry_y == tag] = idx
        
        # Image Load
        for i, path in enumerate(flatten_spt_x):
            spt_x[i] = self.transform(path)
        for i, path in enumerate(flatten_qry_x):
            qry_x[i] = self.transform(path)
        
        # (batch, N*K, ch, h, w), (batch, N*K)
        return spt_x, torch.LongTensor(spt_y), qry_x, torch.LongTensor(qry_y)
    
    def __len__(self):
        return self.batch_size