import  torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Learner(nn.Module):
    """
    학습을 주도하는 모델의 구조를 처리하는 Class
    각 Layer에 대한 가중치 정보를 가지고 있어야 해서 다르게 정의됨
    """
    def __init__(self, config):
        super(Learner, self).__init__()
        
        self.config = config # 모델의 Architecure를 담은 변수 ex) ['conv2d', [32, 3, 3, 3, 1, 0]]
        
        self.vars = nn.ParameterList() # 가중치를 저장하는 변수
        self.vars_bn = nn.ParameterList() # batch_norm 가중치를 저장하는 변수
        
        # config에 입력된 Layer정보대로 가중치를 생성해 저장하는 부분 #
        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                w = nn.Parameter(torch.ones(*param[:4])) # [ch_out, ch_in, kernel_sz, kernel_sz], grad=True 가중치화 함
                torch.nn.init.kaiming_normal_(w) # 가중치 초기화 작업
                
                b = nn.Parameter(torch.zeros(param[0])) # Bias에 대한 가중치 생성 [ch_out]
                
                self.vars.append(w)
                self.vars.append(b)
            elif name == 'convt2d':
                w = nn.Parameter(torch.ones(*param[:4])) # [ch_in, ch_out, kernel_sz, kernel_sz], grad=True 가중치화 함
                torch.nn.init.kaiming_normal_(w)
                
                b = nn.Parameter(torch.zeros(param[1])) # Bias에 대한 가중치 생성 [ch_out]
                
                self.vars.append(w)
                self.vars.append(b)
            elif name == 'linear':
                w = nn.Parameter(torch.ones(*param)) # [ch_out, ch_in]
                torch.nn.init.kaiming_normal_(w)
                
                b = nn.Parameter(torch.zeros(param[0])) # Bias에 대한 가중치 생성 [ch_out]
                
                self.vars.append(w)
                self.vars.append(b)
            elif name == 'bn':
                w = nn.Parameter(torch.ones(param[0]))
                b = nn.Parameter(torch.zeros(param[0]))
                
                self.vars.append(w)
                self.vars.append(b)
                
                # batch_norm은 절반은 가중치 갱신이 이루어지지 않은 연산임
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue;
            else:
                raise NotImplementedError
                
        
    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'
            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'
            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info
    
    
    def forward(self, x, vars=None, bn_training=True):
        """
        순전파 연산을 하는 메소드
        해당 메소드에서 계산하려는 가중치를 지정해줘야한다.(vars)
        None이면 self.vars가 자동으로 매칭되어 계산되며, 
        가중치를 지정하는 경우(fast_weight) 해당 가중치를 매칭하여 계산한다.
        x's shape : [batch, ch, w, h] or [batch, ch]
        """
        if vars is None:
            vars = self.vars
        
        idx = 0     # 가중치에 접근하는 index
        bn_idx = 0  # bn 가중치에 접근하는 index
        for name, param in self.config:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx+1]
                x  = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx+1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name == 'linear':
                w, b = vars[idx], vars[idx+1]
                x = F.linear(x, w, b)
                idx += 2
            elif name == 'bn':
                w, b = vars[idx], vars[idx+1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'flatten':
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                x = x.view(x.size(0), *param) # [b, 8] => [b, 2, 2, 2]
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError
                
        # 제대로 연산했는지 확인
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        
        return x
    
    
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()


    def parameters(self):
        """
        모델의 가중치를 반환하는 메소드를 재정의
        """
        return self.vars