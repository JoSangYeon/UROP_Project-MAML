import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm, notebook

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def evaluate(maml, device, dataset, mode='valid'):
    loader = enumerate(DataLoader(dataset, 1, shuffle=True, num_workers=1, pin_memory=True))
    
    if mode == 'test': 
        loader = tqdm(loader, file=sys.stdout)
    

    losses_all_test = []
    accs_all_test = []
    for idx, (x_spt, y_spt, x_qry, y_qry) in loader:
        x_spt, y_spt, x_qry, y_qry = (
            x_spt.to(device),
            y_spt.to(device),
            x_qry.to(device),
            y_qry.to(device),
        )

        losses, accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
        losses_all_test.append(losses)
        accs_all_test.append(accs)
        if mode == 'test':
            loader.set_postfix(loss='{:.6f}, acc={:.3f}'.format(np.mean(losses), np.mean(accs)))
    if mode == 'test':
        loader.close()

    # [b, update_step+1]
    losses = np.array(losses_all_test).mean(axis=0).astype(np.float16)
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    losses_mean = np.mean(losses)
    accs_mean = np.mean(accs)
    
    return losses.tolist(), accs.tolist(), losses_mean, accs_mean


def train(maml, device, epochs, task_num, train_db, valid_db, train_separater=64, valid_separater=512):
    history = {
        'train_loss' : [],
        'train_acc' : [],
        'train_loss_mean' : [],
        'train_acc_mean' : [],
        'valid_loss' : [],
        'valid_acc' : [],
        'valid_loss_mean' : [],
        'valid_acc_mean' : []
    }

    for epoch in range(epochs):
        # fetch meta_batchsz num of episode each time
        train_loader = DataLoader(train_db, task_num, shuffle=True, num_workers=1, pin_memory=True)
        pbar = tqdm(enumerate(train_loader), file=sys.stdout)

        train_step_history = [[], [], [], []] # loss, acc, loss_mean, acc_mean
        valid_step_history = [[], [], [], []] # loss, acc, loss_mean, acc_mean
        for step, (x_spt, y_spt, x_qry, y_qry) in pbar:

            x_spt, y_spt, x_qry, y_qry = (
                x_spt.to(device),
                y_spt.to(device),
                x_qry.to(device),
                y_qry.to(device),
            )

            losses, accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % train_separater == 0:
                temp = [losses.tolist(), accs.tolist(), np.mean(losses), np.mean(accs)]
                for i, val in enumerate(temp):
                    train_step_history[i].append(val)

            pbar.set_postfix(epoch=f'{epoch+1}/{epochs}', loss='{:.6f}, acc={:.3f}'.format(np.mean(losses), np.mean(accs)))

            if (step % valid_separater == 0 and step != 0) or step == len(train_db)//task_num:  # evaluation
                eval_result = evaluate(maml, device, valid_db)
                for i, val in enumerate(eval_result):
                    valid_step_history[i].append(val)
        pbar.close()
        
        epoch_history = train_step_history + valid_step_history
        for i, key in enumerate(history.keys()):
            history[key].append(epoch_history[i])
            
        print("Mean Train Loss : {:.6f}".format(np.mean(train_step_history[-2])))
        print("Mean Train acc  : {:.4f}".format(np.mean(train_step_history[-1])))
        print("Mean Valid Loss : {:.6f}".format(np.mean(valid_step_history[-2])))
        print("Mean Valid acc  : {:.4f}".format(np.mean(valid_step_history[-1])))
        print()

    return history