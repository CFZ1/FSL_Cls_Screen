from models.base import Trainer
from copy import deepcopy
from utils import count_acc,Averager,count_acc_perClass,save_list_to_txt,count_acc_topk
from .Network import MYNET
import time
import numpy as np
import torch
import torch.nn.functional as F
import os
from terminaltables import AsciiTable


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args,MYNET)

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        #gen_mask
        masknum=3
        mask=np.zeros((args.base_class,args.num_classes))
        for i in range(args.num_classes-args.base_class):
            picked_dummy=np.random.choice(args.base_class,masknum,replace=False)
            mask[:,i+args.base_class][picked_dummy]=1
        mask=torch.tensor(mask).cuda()



        for session in range(args.start_session, args.sessions):
            
            if session == 0:  # load base class train img label
                self.train_s0(base_meta=False,result_list=result_list,base_train_args=mask)
            else:  # incremental learning sessions
                #save dummy classifiers
                self.dummy_classifiers=deepcopy(self.model.module.fc.weight.detach())
                
                self.dummy_classifiers=F.normalize(self.dummy_classifiers[self.args.base_class:,:],p=2,dim=-1)
                self.old_classifiers=self.dummy_classifiers[:self.args.base_class,:]
                self.train_s1_manyRuns(result_list=result_list)

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
    def base_train(self, model, trainloader, optimizer, scheduler, epoch, args,mask):
        tl = Averager()
        tl_mask = Averager()
        tl_cls = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        model = model.train()
    
        for i, batch in enumerate(trainloader, 1):
    
            beta=torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()
            data, train_label = [_.cuda() for _ in batch]
            
            embeddings,_=model.module.encode(data)
    
            logits,mask_sigmoid  = model(data)
            logits_ = logits[:, :args.base_class]
            loss = F.cross_entropy(logits_, train_label)
             #---Different from 01base_episode---6
            tl_cls.add(loss.item(),logits_.size(0))
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
            
            if epoch>=args.loss_iter:
                logits_masked = logits.masked_fill(F.one_hot(train_label, num_classes=model.module.pre_allocate) == 1, -1e9)
                logits_masked_chosen= logits_masked * mask[train_label]
                pseudo_label = torch.argmax(logits_masked_chosen[:,args.base_class:], dim=-1) + args.base_class
                #pseudo_label = torch.argmax(logits_masked[:,args.base_class:], dim=-1) + args.base_class
                loss2 = F.cross_entropy(logits_masked, pseudo_label)
    
                index = torch.randperm(data.size(0)).cuda()
                pre_emb1=model.module.pre_encode(data)
                mixed_data=beta*pre_emb1+(1-beta)*pre_emb1[index]
                mixed_logits=model.module.post_encode(mixed_data)
    
                newys=train_label[index]
                idx_chosen=newys!=train_label
                mixed_logits=mixed_logits[idx_chosen]
    
                pseudo_label1 = torch.argmax(mixed_logits[:,args.base_class:], dim=-1) + args.base_class # new class label
                pseudo_label2 = torch.argmax(mixed_logits[:,:args.base_class], dim=-1)  # old class label
                loss3 = F.cross_entropy(mixed_logits, pseudo_label1)
                novel_logits_masked = mixed_logits.masked_fill(F.one_hot(pseudo_label1, num_classes=model.module.pre_allocate) == 1, -1e9)
                loss4 = F.cross_entropy(novel_logits_masked, pseudo_label2)
                total_loss = loss+args.balance*(loss2+loss3+loss4)
            else:
                total_loss = loss
    
            tl.add(total_loss.item(),logits_.size(0))
            lgt=torch.cat([lgt,logits_[:,:args.base_class].cpu()])
            lbs=torch.cat([lbs,train_label.cpu()])
    
            optimizer.zero_grad()
            #loss.backward()
            total_loss.backward()
            optimizer.step()
        tl = tl.item()
        tl_cls = tl_cls.item()
        if args.procedure in ['singleLabel1']:
            ta = count_acc(lgt, lbs)
            ta_cls,_ = count_acc_perClass(lgt, lbs)
            if args.debug3_maskLoss:
                tl_mask = tl_mask.item()
                print('--------epo {}, train, loss={:.4f} loss_cls={:.4f} tl_mask={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch,tl,tl_cls,tl_mask,ta,ta_cls))
            else:
                print('--------epo {}, train, loss={:.4f} loss_cls={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch,tl,tl_cls,ta,ta_cls))
        return tl, ta
    
    def test(self, model, testloader, epoch,args, session,validation=True, mode_test='test',result_list=None):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                logits,_ = model(data)
                logits = logits[:, :test_class]
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

    def test_intergrate(self, model, testloader, epoch,args, session,validation=True, mode_test='test'):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])

        proj_matrix=torch.mm(self.dummy_classifiers,F.normalize(torch.transpose(model.module.fc.weight[:test_class, :],1,0),p=2,dim=-1))
        
        eta=args.eta
#         print('---eta-----',eta,'---proj_matrix--',proj_matrix)
        
        softmaxed_proj_matrix=F.softmax(proj_matrix,dim=1)

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                
                emb,_=model.module.encode(data)
            
                proj=torch.mm(F.normalize(emb,p=2,dim=-1),torch.transpose(self.dummy_classifiers,1,0))
                # topk, indices = torch.topk(proj, 40)
                #--------------work when novel_class>topk的第二个参数，proj--->res_logit: 即保留前k个数值，其他置为0
                k = args.way
                topk, indices = torch.topk(proj, k)
                res = (torch.zeros_like(proj))
                res_logit = res.scatter(1, indices, topk)
                #--------------work when novel_class>1

                logits1=torch.mm(res_logit,proj_matrix)
                logits2 = model.module.forpass_fc(data)[:, :test_class] 
                logits=eta*F.softmax(logits1,dim=1)+(1-eta)*F.softmax(logits2,dim=1)
#                 print('-----------eta_logits1---',logits1[0,:])
            
                loss = F.cross_entropy(logits, test_label)
                vl.add(loss.item(),logits.size(0))
                lgt=torch.cat([lgt,logits.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
            vl = vl.item()
            if args.procedure in ['singleLabel1']:
                va = count_acc(lgt, lbs)
                va_cls,va_cls_perCls = count_acc_perClass(lgt, lbs)
                print('epo {}, {}, loss={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch, mode_test, vl, va,va_cls))
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
                    results_print["result"] =tuple( ['{va_int:0.3f}%'.format(va_int=va_cls_perCls[ind]*100) for ind in range(lgt.shape[-1])] )
                    table_data = [results_print["name"],results_print["result"]]
                    table = AsciiTable(table_data)
                    print(table.table)
                if session>0:
                    ta_np = np.array(va_cls_perCls)
                    print('Session1 (base/novel/all): {:.3f}/{:.3f}/{:.3f}'.format(ta_np[:args.base_class].mean()*100,ta_np[args.base_class:test_class].mean()*100,ta_np[:test_class].mean()*100))           
        return vl, va, va_cls_perCls
