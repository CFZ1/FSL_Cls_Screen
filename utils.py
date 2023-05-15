import random
import torch
import os
import time
import numpy as np
import pprint as pprint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import matplotlib
_utils_pp = pprint.PrettyPrinter()

def count_acc_perClass(lgt, lbs):
    va_list = []
    for i in range(lgt.shape[-1]):
        if len(lbs[lbs==i]) > 0: 
            va_list.append( count_acc(lgt[lbs==i], lbs[lbs==i]))
        else:
            va_list.append(-1)
            print('count_acc_perClass: There may be a mistake')
    return np.mean(va_list),va_list

def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path,'\n')
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    # def add(self, x):
    #     self.v = (self.v * self.n + x) / (self.n + 1)
    #     self.n += 1
    def add(self, x,num):
        if num>0:
            self.v = (self.v * self.n + x*num) / (self.n + num)
            self.n += num

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def count_acc_topk(x,y,k=5):
    _,maxk = torch.topk(x,k,dim=-1)
    total = y.size(0)
    test_labels = y.view(-1,1) 
    #top1=(test_labels == maxk[:,0:1]).sum().item()
    topk=(test_labels == maxk).sum().item()
    return float(topk/total)

def count_acc_taskIL(logits, label,args):
    basenum=args.base_class
    incrementnum=(args.num_classes-args.base_class)/args.way
    for i in range(len(label)):
        currentlabel=label[i]
        if currentlabel<basenum:
            logits[i,basenum:]=-1e9
        else:
            space=int((currentlabel-basenum)/args.way)
            low=basenum+space*args.way
            high=low+args.way
            logits[i,:low]=-1e9
            logits[i,high:]=-1e9

    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def confmatrix(logits,label,filename):
    
    font={'family':'FreeSerif','size':18}
    matplotlib.rc('font',**font)
    matplotlib.rcParams.update({'font.family':'FreeSerif','font.size':18})
    plt.rcParams["font.family"]="FreeSerif"

    pred = torch.argmax(logits, dim=1)
    cm=confusion_matrix(label, pred,normalize='true')
    #print(cm)
    clss=len(cm)
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    cax = ax.imshow(cm,cmap=plt.cm.jet) 
    if clss<=100:
        plt.yticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=16)
        plt.xticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=16)
    elif clss<=200:
        plt.yticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=16)
        plt.xticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=16)
    else:
        plt.yticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=16)
        plt.xticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=16)

    plt.xlabel('Predicted Label',fontsize=20)
    plt.ylabel('True Label',fontsize=20)
    plt.tight_layout()
    plt.savefig(filename+'.pdf',bbox_inches='tight')
    plt.close()

    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    cax = ax.imshow(cm,cmap=plt.cm.jet) 
    cbar = plt.colorbar(cax) # This line includes the color bar
    cbar.ax.tick_params(labelsize=16)
    if clss<=100:
        plt.yticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=16)
        plt.xticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=16)
    elif clss<=200:
        plt.yticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=16)
        plt.xticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=16)
    else:
        plt.yticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=16)
        plt.xticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=16)
    plt.xlabel('Predicted Label',fontsize=20)
    plt.ylabel('True Label',fontsize=20)
    plt.tight_layout()
    plt.savefig(filename+'_cbar.pdf',bbox_inches='tight')
    plt.close()

    return cm





def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()
    

def average_performance(pred, target, thr=0.5, k=None,per_class = False):
    """Calculate CP, CR, CF1, OP, OR, OF1, where C stands for per-class
    average, O stands for overall average, P stands for precision, R stands for
    recall and F1 stands for F1-score.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
        thr (float): The confidence threshold. Defaults to None.
        k (int): Top-k performance. Note that if thr and k are both given, k
            will be ignored. Defaults to None.

    Returns:
        tuple: (CP, CR, CF1, OP, OR, OF1)
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')
    if thr is None and k is None:
        thr = 0.5

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'

    eps = np.finfo(np.float32).eps
    target[target == -1] = 0
    if thr is not None:
        # a label is predicted positive if the confidence is no lower than thr
        pos_inds = pred >= thr

    else:
        # top-k labels will be predicted positive for any example
        sort_inds = np.argsort(-pred, axis=1)
        sort_inds_ = sort_inds[:, :k]
        inds = np.indices(sort_inds_.shape)
        pos_inds = np.zeros_like(pred)
        pos_inds[inds[0], sort_inds_] = 1

    tp = (pos_inds * target) == 1
    fp = (pos_inds * (1 - target)) == 1
    fn = ((1 - pos_inds) * target) == 1
    
    acc_class = 1.-(fp.sum(axis=0)+fn.sum(axis=0))/pred.shape[0] #-----------new
    precision_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fn.sum(axis=0), eps)
    # CA =  acc_class.mean() * 100.0
    # CP = precision_class.mean() * 100.0
    # CR = recall_class.mean() * 100.0
    CA =  acc_class.mean()
    CP = precision_class.mean()
    CR = recall_class.mean()
    CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)
    # OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
    # OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
    OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps)
    OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps)
    OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)
    # return CA, CP, CR, CF1, OP, OR, OF1
    if per_class:
        return acc_class,precision_class,recall_class
    else:
        return CA, CP, CR, CF1


