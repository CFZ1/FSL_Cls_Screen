from models.base import Trainer
from copy import deepcopy
from utils import count_acc,Averager,count_acc_perClass,save_list_to_txt
from .Network import MYNET
import time
import numpy as np
import torch
import torch.nn.functional as F
import os
from terminaltables import AsciiTable
import random


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args,MYNET)
        
    def train(self):
        
        args = self.args
        t_start_time = time.time()

        result_list = [args]

        for session in range(args.start_session, args.sessions):
            self.session = session  #------------230402
            if session == 0:
                self.train_s0(base_meta=True,result_list=result_list) #base_meta=True: episode train
            else:
                self.pred_reslut = [] #------------230402
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
        tl_cls = Averager()
        tl_mask = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        model = model.train()
        
        if args.debug3_maskLoss:#------------------------1215
            normal_id = trainloader.dataset.base_cls.index('normal')
        k_support = args.episode_way * args.episode_shot 
        for i, batch in enumerate(trainloader, 1):
            data, train_label = [_.cuda() for _ in batch]
    
            #---Different from 01base---2
            model.module.mode = 'encoder'
            img_feas,mask_sigmoid = model(data) #---Different from 01base_episode---s1
            proto_feas, query_feas = img_feas[:k_support], img_feas[k_support:]
            model.module.mode = args.base_mode
            if args.debug3_maskLoss: #------------------------1215
                normal_ids = torch.where(train_label==normal_id)[0] 
            if args.episode_way != args.base_class: 
                train_label = torch.arange(args.episode_way).repeat(args.episode_shot+args.episode_query).to(device=train_label.device)
            proto_label = torch.arange(0,args.episode_way)
            protos,proto_list_mask = model.module.forward_proto(proto_feas,train_label[:k_support],proto_label)
            #------------TODO,每一个`epoch的base_train`之后的test使用的都是最后的model.module.fc.weight和model.module.protos_mask
            model.module.fc.weight.data[int(min(train_label)):int(max(train_label))+1] = protos
            if self.args.debug2_relationMask:
                model.module.protos_mask[int(min(train_label)):int(max(train_label))+1] = proto_list_mask   
            #------------TODO,
            logits = model.module.forward_logits(query_feas,proto_list_mask)
            #---Different from 01base---2
            
            logits = logits[:, :args.episode_way]#------------------------1124
            loss_cls = F.cross_entropy(logits, train_label[k_support:]) #---Different from 01base_episode---5
            
            #---Different from 01base_episode---6
            if args.debug3_maskLoss:
                if len(normal_ids)==0:
                    loss_mask = 0.0
#                     print('---loss_mask---',0.0)
                else:
                    k_normal_ids = normal_ids[:args.episode_shot] #support_normal#------------------------1124
                    normal_mask = mask_sigmoid[k_normal_ids] #-------------220918 
                    loss_mask = F.binary_cross_entropy(normal_mask,torch.zeros_like(normal_mask).detach())
                    tl_mask.add(loss_mask.item(),k_normal_ids.size(0))
                loss = loss_cls + args.mask_weight*loss_mask
#                 print('---loss_mask---',loss_mask.item())
                tl_cls.add(loss_cls.item(),logits.size(0))
            else:
                loss = loss_cls
            #---Different from 01base_episode---6
                
            tl.add(loss.item(),logits.size(0))
            lgt=torch.cat([lgt,logits[:,:args.base_class].cpu()])
            lbs=torch.cat([lbs,train_label[k_support:].cpu()])
    
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tl = tl.item()
        tl_mask = tl_mask.item()
        tl_cls = tl_cls.item()
        if args.procedure in ['singleLabel1']:
            ta = count_acc(lgt, lbs)
            ta_cls,_ = count_acc_perClass(lgt, lbs)
            if args.debug3_maskLoss:
                print('--------epo {}, train, loss={:.4f} loss_cls={:.4f} loss_mask={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch, tl,tl_cls,tl_mask,ta,ta_cls))
            else:
                print('--------epo {}, train, loss={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch, tl,ta,ta_cls))
            if args.debug1_relationAdd:
                print('relation_lambda:', model.module.relation_lambda)
#             if args.debug2_relationMask:
#                 print('relation_mask:', model.module.relation_mask)
        return tl, ta
    def test(self,model, testloader, epoch,args, session,validation=True, mode_test='test',result_list=None):
        test_class = args.base_class + session * args.way
        model = model.eval()
        tl1 = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])
#         data_embedding_s=torch.tensor([])  #------------230402
        if args.debug2_relationMask:
            print('protos_mask: ', model.module.protos_mask[:,0])
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                logits = model(data)
                logits = logits[:, :test_class]
                loss = F.cross_entropy(logits, test_label)
#                 data_embedding_,_ = model.module.encode(data) #------------230402
#                 data_embedding_s = torch.cat([data_embedding_s,data_embedding_.cpu()]) #------------230402
    
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
                torch.save({'img_names':testloader.dataset.data, 'logits':lgt,'labels':lbs}, save_tensor_pth)
                if self.session ==1: #------------230402
#                     self.pred_reslut.append({'classifier_weights':model.module.fc.weight.data,
#                         'protos_mask':model.module.protos_mask,
#                         'data_embedding':data_embedding_s,
#                         'img_names':testloader.dataset.data,'logits':lgt,'labels':lbs})
                    self.pred_reslut.append({'logits':lgt,'labels':lbs}) #------------230402
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
        
    def replace_base_fc(self,trainset, transform, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
    
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base,
                                                  num_workers=args.num_workers, pin_memory=True, shuffle=False)
        trainloader.dataset.transform = transform
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                embedding,_ = model(data)
    
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        #---Different from 01base---e3
        proto_label = torch.arange(0,args.base_class)
        proto_list,proto_list_mask = model.module.forward_proto(embedding_list,label_list,proto_label)
        #---Different from 01base---e3
    
        model.module.fc.weight.data[:args.base_class] = proto_list
        if self.args.debug2_relationMask:
            model.module.protos_mask[:args.base_class] = proto_list_mask
    
        return model
