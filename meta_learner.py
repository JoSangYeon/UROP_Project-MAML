import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np

from learner import Learner
from copy import deepcopy


class Meta_Learner(nn.Module):
    """
    MAML의 알고리즘(outer, inner loop)을 구현한 Model 구조 Class
    """
    def __init__(self, args, config, task='classification'):
        super(Meta_Learner, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config)
        self.meta_optim = optim.AdamW(self.net.parameters(), lr=self.meta_lr)

        # Loss_fn 정의
        self.task = task

    def criterion(self, logits, label):
        if self.task == 'classification':
            return F.cross_entropy(logits, label)
        else:
            return F.mse_loss(logits, label)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        loss 소실, 폭발을 방지하기 위한 메소드
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter
    
    
    def inner_optim(self, grad, weights):
        gradient_descent = lambda p: p[1] - self.update_lr * p[0]
        return_weights = list(map(gradient_descent, zip(grad, weights)))
        return return_weights


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        MAML의 inner, outter loop을 구현한 순전파 식 정의 메소드
        b = task_num
        setsz, querysz = n-way k-shot -> n*k
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:        loss, acc
        """
        task_num = x_spt.size(0)
        querysz = x_qry.size(1) # support set의 각 배치마다의 개수(N-way K-shot이면 N*K개)

        losses_q = [0 for _ in range(self.update_step+1)] # [update전 loss, 1번 갱신 후 loss, ,,, , n번 갱신 후 loss]
        corrects = [0 for _ in range(self.update_step+1)] # [update전 acc, 1번 갱신 후 acc, n번 갱신 후 acc]

        for i in range(task_num): # task_num만큼 Sampling된 TASK가 innerloop에 수행됨
            # 1. i번째 Task에 대해 첫번째 가중치를 갱신
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = self.criterion(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters()) # 미분값 구하기
            base_weights = self.inner_optim(grad, self.net.parameters()) # base가 되는 weights, 경사하강법

            # 가중치를 갱신하기 전 Loss, acc(losses_q[0], corrects[0])
            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = self.criterion(logits_q, y_qry[i])
                losses_q[0] += loss_q

                if self.task == 'classification':
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] += correct

            # 첫번째 가중치 갱신 후의 Loss, acc(losses_q[1], corrects[1])
            with torch.no_grad():
                logits_q = self.net(x_qry[i], base_weights, bn_training=True)
                loss_q = self.criterion(logits_q, y_qry[i])
                losses_q[1] += loss_q

                if self.task == 'classification':
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] += correct

            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], base_weights, bn_training=True)
                loss = self.criterion(logits, y_spt[i])
                grad = torch.autograd.grad(loss, base_weights)
                base_weights = self.inner_optim(grad, base_weights)

                logits_q = self.net(x_qry[i], base_weights, bn_training=True)
                loss_q = self.criterion(logits_q, y_qry[i])
                # sampling된 task들에 의해 losses_q는 sum되고 최종적으로 마지막 갱신 loss에서 self.net의 가중치가 갱신된다.
                losses_q[k+1] += loss_q
                
                if self.task == 'classification':
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[k+1] += correct
        
        loss_q = losses_q[-1] / task_num # Query set에 대한 총 Loss에 대해서 평균을 구함
        self.meta_optim.zero_grad()      # 가중치 갱신값 0으로 초기화
        loss_q.backward()                # 역전파를 통해 미분값 개산
        self.meta_optim.step()           # 가중치 갱신
        
        losses = np.array(list(map(lambda x: x.item()/task_num , losses_q)))
        accs = np.array(corrects) / (querysz * task_num)
        
        return losses, accs

    
    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        Meta-Test dataset에 대해 수행하는 작업
        batch가 1이여서 squeeze(0)해서 batch가 없앰
        :param x_spt:   [1, setsz, c_, h, w]
        :param y_spt:   [1, setsz]
        :param x_qry:   [1, querysz, c_, h, w]
        :param y_qry:   [1, querysz]
        :return:        loss, acc
        """
        assert x_spt.shape[0] == 1
        x_spt = x_spt.squeeze(0); y_spt = y_spt.squeeze(0)
        x_qry = x_qry.squeeze(0); y_qry = y_qry.squeeze(0)
        
        querysz = x_qry.size(0) # Query set에 대해서 N-way K-shot일때, N*K의 값
        
        losses = [0 for _ in range(self.update_step_test+1)]
        corrects = [0 for _ in range(self.update_step_test+1)]
        
        # 연산할때 bn의 가중치가 변경될 수 있어서 모델을 그대로 복사해서 사용
        net = deepcopy(self.net)
        
        logits = net(x_spt)
        loss = self.criterion(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        base_weights = self.inner_optim(grad, net.parameters())
        
        # 가중치를 갱신하기 전 Loss, acc(losses[0], corrects[0])
        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            loss_q = self.criterion(logits_q, y_qry)
            losses[0] += loss_q.item()
            
            if self.task == 'classification':
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[0] += correct
                
        # 첫번째 가중치 갱신 후의 Loss, acc(losses[1], corrects[1])    
        with torch.no_grad():
            logits_q = net(x_qry, base_weights, bn_training=True)
            loss_q = self.criterion(logits_q, y_qry)
            losses[1] += loss_q.item()
            
            if self.task == 'classification':
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[1] += correct
                
        for k in range(1, self.update_step_test):
            logits = net(x_spt, base_weights, bn_training=True)
            loss = self.criterion(logits, y_spt)
            grad = torch.autograd.grad(loss, base_weights)
            base_weights = self.inner_optim(grad, base_weights)

            logits_q = net(x_qry, base_weights, bn_training=True)
            loss_q = self.criterion(logits_q, y_qry)
            losses[k+1] += loss_q.item()
            
            if self.task == 'classification':
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry).sum().item()
                    corrects[k+1] += correct
        
        del net
        
        losses = np.array(losses)
        accs = np.array(corrects) / querysz
        
        return losses, accs
    
    def inference(self, x):
        """
        parm x: [batch, shpae~]
                [batch, ch, h, w]
                [batch, ch] ...
        return: inference result
        """
        x = self.net(x)
        return x