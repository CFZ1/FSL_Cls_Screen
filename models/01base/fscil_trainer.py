from models.base import Trainer
from utils import count_acc,Averager,count_acc_perClass,save_list_to_txt
from .Network import MYNET
import time
import torch
import torch.nn.functional as F
import os
from terminaltables import AsciiTable
import numpy as np


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args,MYNET)

    def train(self):
        args = self.args
        t_start_time = time.time()

        result_list = [args]

        for session in range(args.start_session, args.sessions):
            if session == 0:
                self.train_s0(base_meta=False,result_list=result_list)
            else:
                self.train_s1_manyRuns(result_list=result_list)
                
        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)         
        
    def base_train(self,model, trainloader, optimizer, scheduler, epoch, args, other_args = None):
        tl = Averager()
        tl_mask = Averager()#-----------------------debug3_maskLoss_11
        tl_cls = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        model = model.train()
    
        for i, batch in enumerate(trainloader, 1):
            data, train_label = [_.cuda() for _ in batch]
    
            logits,mask_sigmoid = model(data) #-----------------------debug3_maskLoss_8
            logits = logits[:, :args.base_class]
            loss = F.cross_entropy(logits, train_label)
            #---Different from 01base_episode---6
            tl_cls.add(loss.item(),logits.size(0))
            if args.debug3_maskLoss: #-----------------------debug3_maskLoss_10
                normal_id = trainloader.dataset.base_cls.index('normal')
                k_normal_ids = torch.where(train_label==normal_id)[0]
                if len(k_normal_ids) !=0:
                    normal_mask = mask_sigmoid[k_normal_ids] #-------------220918 
                    loss_mask = F.binary_cross_entropy(normal_mask,torch.zeros_like(normal_mask).detach())
                    loss = loss + args.mask_weight*loss_mask
    #                 print('---loss_mask---',loss_mask.item())
                    tl_mask.add(loss_mask.item(),len(k_normal_ids))
            #---Different from 01base_episode---6     
    
            total_loss = loss
    
            tl.add(total_loss.item(),logits.size(0))
            lgt=torch.cat([lgt,logits[:,:args.base_class].cpu()])
            lbs=torch.cat([lbs,train_label.cpu()])
    
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tl = tl.item()
        tl_cls = tl_cls.item()#-----------------------debug3_maskLoss_12
        if args.procedure in ['singleLabel1']:
            ta = count_acc(lgt, lbs)
            ta_cls,_ = count_acc_perClass(lgt, lbs)
            if args.debug3_maskLoss:#-----------------------debug3_maskLoss_13
                tl_mask = tl_mask.item()
                print('--------epo {}, train, loss={:.4f} loss_cls={:.4f} tl_mask={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch,tl,tl_cls,tl_mask,ta,ta_cls))
            else:
                print('--------epo {}, train, loss={:.4f} loss_cls={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch,tl,tl_cls,ta,ta_cls))
        return tl, ta
    def test(self,model, testloader, epoch,args, session,validation=True, mode_test='test',result_list=None):
        test_class = args.base_class + session * args.way
        model = model.eval()
        tl1 = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                logits,_ = model(data)#-----------------------debug3_maskLoss_14
                logits = logits[:, :test_class]
                loss = F.cross_entropy(logits, test_label)
    
                tl1.add(loss.item(),logits.size(0))
    
                lgt=torch.cat([lgt,logits.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
            tl1 = tl1.item()
            if args.procedure in ['singleLabel1']:
                ta = count_acc(lgt, lbs)
                ta_cls,ta_cls_perCls = count_acc_perClass(lgt, lbs)
                print('epo {}, {}, loss={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch, mode_test, tl1, ta,ta_cls))
            
            if validation is not True:
                save_tensor_pth = os.path.join(args.save_path, 'session' + str(session) + 'result.pth')
                torch.save({'img_names':testloader.dataset.data,'logits':lgt,'labels':lbs}, save_tensor_pth)
                if args.procedure in ['singleLabel1']:
                    #---print---
                    results_print = {}
                    if session==0:
                        results_print["name"] = tuple(testloader.dataset.base_cls)
                    else:
                        results_print["name"] = tuple(testloader.dataset.base_cls+testloader.dataset.novel_cls)
                    results_print["result"] =tuple( ['{va_int:0.3f}%'.format(va_int=ta_cls_perCls[ind]*100) for ind in range(lgt.shape[-1])] )
                    table_data = [results_print["name"],results_print["result"]]
                    table = AsciiTable(table_data)
                    print(table.table)
                    if result_list is not None:
                        result_list.append(table.table)
                if session>0:
                    ta_np = np.array(ta_cls_perCls)
                    print('Session1 (base/novel/all): {:.3f}/{:.3f}/{:.3f}'.format(ta_np[:args.base_class].mean()*100,ta_np[args.base_class:test_class].mean()*100,ta_np[:test_class].mean()*100))
        if ta_cls is not None:
            return tl1, ta, ta_cls_perCls
        else:
            return tl1, ta

