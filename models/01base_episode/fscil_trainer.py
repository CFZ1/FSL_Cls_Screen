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
                #-----------------------Different from 01base------------------s1
                self.train_s0(base_meta=True,result_list=result_list)
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
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        model = model.train()
    
        for i, batch in enumerate(trainloader, 1):
            data, train_label = [_.cuda() for _ in batch]
    
            #---Different from 01base---2
            k_support = args.episode_way * args.episode_shot 
            model.module.mode = 'encoder'
            img_feas = model(data)
            proto_feas, query_feas = img_feas[:k_support], img_feas[k_support:]
            model.module.mode = args.base_mode
            proto_label = torch.arange(0,args.base_class) 
            protos = model.module.forward_proto(proto_feas,train_label[:k_support],proto_label)
            model.module.fc.weight.data[int(min(train_label)):int(max(train_label))+1] = protos
            logits = model.module.forward_logits(query_feas)
            train_label = train_label[k_support:]
            #---Different from 01base---2
            logits = logits[:, :args.base_class]
            loss = F.cross_entropy(logits, train_label)
    
    
            total_loss = loss
    
            tl.add(total_loss.item(),logits.size(0))
            lgt=torch.cat([lgt,logits[:,:args.base_class].cpu()])
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
                embedding = model(data)
    
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        #---Different from 01base---3
        proto_label = torch.arange(0,args.base_class)
        proto_list = model.module.forward_proto(embedding_list,label_list,proto_label)
        #---Different from 01base---3    

    
        model.module.fc.weight.data[:args.base_class] = proto_list
    
        return model

