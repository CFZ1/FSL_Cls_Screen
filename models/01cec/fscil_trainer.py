from models.base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from utils import count_acc,Averager,count_acc_perClass,save_list_to_txt
from dataloader.data_utils import set_up_datasets
from .Network import MYNET
import torch
import torch.nn.functional as F
import random
import time
import os
from terminaltables import AsciiTable
import numpy as np


class FSCILTrainer(Trainer):
    def __init__(self, args):
        if args.model_dir == None:
            print('*********WARNINGl: NO INIT MODEL**********')
        args.not_data_init = False
        super().__init__(args,MYNET)

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        return model_dict

    def replace_to_rotate(self, proto_tmp, query_tmp):
        for i in range(self.args.low_way):
            # random choose rotate degree,逆时针旋转
            rot_list = [90, 180, 270]
            sel_rot = random.choice(rot_list)
            if sel_rot == 90:  # rotate 90 degree
                # print('rotate 90 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
            elif sel_rot == 180:  # rotate 180 degree
                # print('rotate 180 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].flip(2).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].flip(2).flip(3)
            elif sel_rot == 270:  # rotate 270 degree
                # print('rotate 270 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
        return proto_tmp, query_tmp

    def get_optimizer_base(self):
        if self.args.debug3_maskLoss: #-----------------------debug3_maskLoss_3e
            optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                         {'params': self.model.module.conv_mask.parameters(), 'lr': self.args.lr_base},
                                     {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)           
        else:
            optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                     {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)
        return optimizer, scheduler

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):
            if session == 0:  # load base class train img label
                self.best_model_dict = self.update_param(self.model, self.best_model_dict)
                self.train_s0(base_meta=True,result_list=result_list) #base_meta=True: episode train
            else:  # incremental learning sessions
                self.train_s1_manyRuns(result_list=result_list)

        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        result_list.append('Best epoch:%d' % self.trlog['max_acc_epoch'])
        print('Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

    def base_train(self, model, trainloader, optimizer, scheduler, epoch, args, other_args = None):
        tl = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])

        label = torch.arange(args.episode_way + args.low_way).repeat(args.episode_query)
        label = label.type(torch.cuda.LongTensor)

        for i, batch in enumerate(trainloader, 1):
            data, true_label = [_.cuda() for _ in batch]

            k = args.episode_way * args.episode_shot
            proto, query = data[:k], data[k:]
            # sample low_way data
            proto_tmp = deepcopy(
                proto.reshape(args.episode_shot, args.episode_way, proto.shape[1], proto.shape[2], proto.shape[3])[
                :args.low_shot,
                :args.low_way, :, :, :].flatten(0, 1))
            query_tmp = deepcopy(
                query.reshape(args.episode_query, args.episode_way, query.shape[1], query.shape[2], query.shape[3])[:,
                :args.low_way, :, :, :].flatten(0, 1))
            # random choose rotate degree
            proto_tmp, query_tmp = self.replace_to_rotate(proto_tmp, query_tmp)

            model.module.mode = 'encoder'
            data = model(data)
            proto_tmp = model(proto_tmp)
            query_tmp = model(query_tmp)

            # k = args.episode_way * args.episode_shot
            proto, query = data[:k], data[k:]

            proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])
            query = query.view(args.episode_query, args.episode_way, query.shape[-1])

            proto_tmp = proto_tmp.view(args.low_shot, args.low_way, proto.shape[-1])
            query_tmp = query_tmp.view(args.episode_query, args.low_way, query.shape[-1])

            proto = proto.mean(0).unsqueeze(0)
            proto_tmp = proto_tmp.mean(0).unsqueeze(0)

            proto = torch.cat([proto, proto_tmp], dim=1)
            query = torch.cat([query, query_tmp], dim=1)

            proto = proto.unsqueeze(0)
            query = query.unsqueeze(0)

            logits = model.module._forward(proto, query)

            total_loss = F.cross_entropy(logits, label)

            tl.add(total_loss.item(),logits.size(0))
            lgt=torch.cat([lgt,logits.cpu()])
            lbs=torch.cat([lbs,label.cpu()])

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        tl = tl.item()
        if args.procedure in ['singleLabel1']:
            ta = count_acc(lgt, lbs)
            ta_cls,_ = count_acc_perClass(lgt, lbs)
            print('--------epo {}, train, loss={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch, tl,ta,ta_cls))
        return tl, ta

    def test(self, model, testloader, epoch, args, session, validation=True, mode_test='test',result_list=None):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)

                proto = model.module.fc.weight[:test_class, :].detach()
                proto = proto.unsqueeze(0).unsqueeze(0)

                logits = model.module._forward(proto, query)

                loss = F.cross_entropy(logits, test_label)
                
                vl.add(loss.item(),logits.size(0)) 
                lgt=torch.cat([lgt,logits.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
            vl = vl.item()
            if args.procedure in ['singleLabel1']:
                ta = count_acc(lgt, lbs)
                ta_cls,ta_cls_perCls = count_acc_perClass(lgt, lbs)
                print('epo {}, {}, loss={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch, mode_test, vl, ta,ta_cls))
            
            if validation is not True:
                save_tensor_pth = os.path.join(args.save_path, 'session' + str(session) + 'result.pth')
                torch.save({'logits':lgt,'labels':lbs}, save_tensor_pth)
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
        return vl, ta, ta_cls_perCls
 
    def validation(self):
        with torch.no_grad():
            model = self.model

            for session in range(1, self.args.sessions):
                train_set, trainloader, testloader = self.get_dataloader(session)

                trainloader.dataset.transform = testloader.dataset.transform
                model.module.mode = 'avg_cos'
                model.eval()
                model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                vl, va = self.test(model, testloader, self.args, session)

        return vl, va

