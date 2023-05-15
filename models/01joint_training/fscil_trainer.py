from models.base import Trainer
from copy import deepcopy

from utils import count_acc,Averager,count_acc_perClass,save_list_to_txt
from .Network import MYNET
import random
import torch
import time
import os
import torch.nn.functional as F
from terminaltables import AsciiTable
import numpy as np


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args, MYNET)
        self.args.sessions = 1
        self.args.not_data_init = True

    
    def get_dataloader(self, session, base_meta = False):
        #------0407
        if self.args.procedure in ['multiLabel','singleLabel1']:
            # NEU和mobilescreen不同，NEU的novel类别已经事先划分为train+test了
            if ('mobilescreen' in self.args.dataset) or ('Magnetic' in self.args.dataset):
                base_trainset, base_valset, novel_baseTestset = self.args.Dataset.get_data_joint_training(self.args)
                train_idx_list = []
                novel_idx_lists = novel_baseTestset.get_novel_ids()
                for novel_idx_list in novel_idx_lists:
                    train_idx_list.extend(random.sample(novel_idx_list, self.args.shot))
                test_idx_list = [i for i in list(range(0,len(novel_baseTestset))) if i not in train_idx_list]
                test_set = deepcopy(novel_baseTestset)
                base_trainset.data.extend([novel_baseTestset.data[i] for i in train_idx_list])
                base_trainset.targets.extend([novel_baseTestset.targets[i] for i in train_idx_list])
                test_set.data = [test_set.data[i] for i in test_idx_list]
                test_set.targets = [test_set.targets[i] for i in test_idx_list]
                
                trainloader = torch.utils.data.DataLoader(dataset=base_trainset, batch_size=self.args.batch_size_base, shuffle=True,
                                                  num_workers=self.args.num_workers, pin_memory=True)
                valloader = torch.utils.data.DataLoader(
                    dataset=base_valset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
                testloader = torch.utils.data.DataLoader(
                    dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
#                 print('train_idx_list',train_idx_list)
            elif 'NEU' in self.args.dataset:
                base_trainset, base_valset, novelBase_testset, novel_trainset = self.args.Dataset.get_data_joint_training(self.args)
                
                train_idx_list = []
                novel_idx_lists = novel_trainset.get_novel_ids()
                for novel_idx_list in novel_idx_lists:
                    train_idx_list.extend(random.sample(novel_idx_list, self.args.shot))
                base_trainset.data.extend([novel_trainset.data[i] for i in train_idx_list])
                base_trainset.targets.extend([novel_trainset.targets[i] for i in train_idx_list])
                
                trainloader = torch.utils.data.DataLoader(dataset=base_trainset, batch_size=self.args.batch_size_base, shuffle=True,
                                                  num_workers=self.args.num_workers, pin_memory=True)
                valloader = torch.utils.data.DataLoader(
                    dataset=base_valset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
                testloader = torch.utils.data.DataLoader(
                    dataset=novelBase_testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
                
        else:
            print('not implement')
        return base_trainset, trainloader, valloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):
            if session == 0:  # load base class train img label
                ta_cls_perCls_list = []
                self.testloader = None
                self.init_model_dict = deepcopy(self.model.state_dict())
                for run in range(self.args.test_runs):
                    print('-----------------',run,'-----------------')###1031
                    self.best_model_dict = self.init_model_dict
                    self.model.load_state_dict(self.best_model_dict)
                    self.trlog['max_acc'][0] = 0.0
                    
                    self.train_s0(base_meta=False,result_list=result_list)
                    self.model.load_state_dict(self.best_model_dict)
                    tsl_test, tsa_test, ta_cls_perCls_test = self.test(self.model, self.testloader, 0, self.args, session,validation=False,mode_test='test')
                    ta_cls_perCls_list.append(ta_cls_perCls_test)
                    
                ta_cls_perCls_ = np.array(ta_cls_perCls_list)
                ta_cls_perCls_ = ta_cls_perCls_.mean(0)
                #--------------------打印100次结果的均值-----------
                results_print = {}
                results_print["name"] = tuple(self.testloader.dataset.base_cls+self.testloader.dataset.novel_cls)
                results_print["result"] =tuple( ['{va_int:0.3f}%'.format(va_int=ta_cls_perCls_[ind]*100) for ind in range(ta_cls_perCls_.shape[0])] )
                table_data = [results_print["name"],results_print["result"]]
                table = AsciiTable(table_data)
                print('-------------Mean value of %d runs-------------' % self.args.test_runs)
                print(table.table)
                if result_list is not None:
                    result_list.append(table.table)
                #--------------------打印100次结果的均值-----------
                print('-------------summary-------------')
                print("base classes:", self.testloader.dataset.base_cls)
                print("novel classes:", self.testloader.dataset.novel_cls)
                print('Session0_val: {:.3f} (bast_epoch: {})'.format(self.trlog['max_acc'][0],self.trlog['max_acc_epoch']))
                print('Session0_avg_val: {:.3f}'.format(self.trlog['avg_acc_val']))
                print('Session0_avg_test: {:.3f}'.format(self.trlog['avg_acc_test']))
                print('Final (base/novel/all): {:.3f}/{:.3f}/{:.3f}'.format(ta_cls_perCls_[:len(self.testloader.dataset.base_cls)].mean()*100,ta_cls_perCls_[len(self.testloader.dataset.base_cls):].mean()*100,ta_cls_perCls_.mean()*100))
                print('    ')
                result_list.append('Session0_val: {:.3f} (bast_epoch: {})'.format(self.trlog['max_acc'][0],self.trlog['max_acc_epoch']))
                result_list.append('Session0_avg_val: {:.3f}'.format(self.trlog['avg_acc_val']))
                result_list.append('Session0_avg_test: {:.3f}'.format(self.trlog['avg_acc_test']))
                result_list.append('Final (base/novel/all): {:.3f}/{:.3f}/{:.3f}'.format(ta_cls_perCls_[:len(self.testloader.dataset.base_cls)].mean()*100,ta_cls_perCls_[len(self.testloader.dataset.base_cls):].mean()*100,ta_cls_perCls_.mean()*100))
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
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        model = model.train()
    
        for i, batch in enumerate(trainloader, 1):
            data, train_label = [_.cuda() for _ in batch]
    
            logits = model(data)
            logits = logits[:, :args.num_classes] ############Different from 01base---s1
            loss = F.cross_entropy(logits, train_label)
    
    
            total_loss = loss
    
            tl.add(total_loss.item(),logits.size(0))
            lgt=torch.cat([lgt,logits[:,:args.num_classes].cpu()])############Different from 01base---s2
            lbs=torch.cat([lbs,train_label.cpu()])
    
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tl = tl.item()
        if args.procedure in ['singleLabel1']:
            ta = count_acc(lgt, lbs)
            ta_cls,_ = count_acc_perClass(lgt, lbs)
            print('--------epo {}, train, loss={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch, tl,ta,ta_cls))
        return tl, ta
    def test(self,model, testloader, epoch,args, session,validation=True, mode_test='test',result_list=None):
        ############Different from 01base---3
        if mode_test=='val':
            session = 0
        elif mode_test=='test':
            session = 1
        ############Different from 01base---3
        test_class = args.base_class + session * args.way
        model = model.eval()
        tl1 = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                logits = model(data)
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
                torch.save({'logits':lgt,'labels':lbs}, save_tensor_pth)
                if args.procedure in ['singleLabel1']:
                    #---print---
                    results_print = {}
                    results_print["name"] = tuple((testloader.dataset.base_cls+testloader.dataset.novel_cls)[:test_class])############Different from 01base---e4
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
        
