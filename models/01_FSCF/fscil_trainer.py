from models.base import Trainer

from utils import count_acc,Averager,count_acc_perClass,save_list_to_txt
from .Network import MYNET
import time
import os
import torch
import torch.nn.functional as F
from terminaltables import AsciiTable

class FSCILTrainer(Trainer):
    def __init__(self, args):
        if ('mobilescreen' in args.dataset) or ('NEU' in args.dataset) or ('Magnetic' in args.dataset):
            if 'FSCF' not in args.dataset:
                args.dataset = args.dataset+'_FSCF'
        args.not_data_init = False #base权重保持原样，不使用特征平均; novel使用特征平均(normalize(特征)的平均的normalize),经过实验，感觉base不可能不用平均
        super().__init__(args, MYNET)
        
    def get_optimizer_base(self):
        if self.args.debug3_maskLoss: #-----------------------debug3_maskLoss_6
            optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                         {'params': self.model.module.conv_mask.parameters(), 'lr': self.args.lr_base},
                                         {'params': self.model.module.fc.parameters(), 'lr': self.args.lrg},
                                         {'params': self.model.module.g.parameters(), 'lr': self.args.lrg}],
                                        momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        else:
            optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                         {'params': self.model.module.fc.parameters(), 'lr': self.args.lrg},
                                         {'params': self.model.module.g.parameters(), 'lr': self.args.lrg}],
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

        result_list = [args]

        for session in range(args.start_session, args.sessions):

            if session == 0:  # load base class train img label
                self.train_s0(base_meta=False,result_list=result_list)
            else:  # incremental learning sessions
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
        tl_con = Averager()
        tl_mask = Averager()#-----------------------debug3_maskLoss_10
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        model = model.train()
        
        for i, batch in enumerate(trainloader, 1):
            data1, data2, train_label = [_.cuda() for _ in batch]
    
            out_1,logits1,mask_sigmoid1 = model(data1)#-----------------------debug3_maskLoss_7,8
            out_2,logits2,mask_sigmoid2 = model(data2)
            #-------------------对比损失---------------------
            batch_size = train_label.shape[0]
            out = torch.cat([out_1, out_2], dim=0) # [2*bs, D]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous())) # [2*bs, 2*bs]
            # [2*B, 2*B-1]
            # mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*(B-1)]
            mask_temp = F.one_hot(torch.cat((torch.arange(batch_size, batch_size+batch_size),torch.arange(0, batch_size)))).to(device=sim_matrix.device)+torch.eye(2*batch_size, device=sim_matrix.device)
            mask = (torch.ones_like(sim_matrix) - mask_temp).bool()
            sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1) # [2*B, 2*(B-1)]
            #-----------Hard negative mining
            sigma = int(args.sigma*sim_matrix.shape[-1])
            sim_matrix = torch.sort(sim_matrix,1,descending=True)[0][:,:sigma]
               #-----------Hard negative mining
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1)) #[bs,]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0) #[2*bs,]
            loss_CT = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            #-------------------对比损失---------------------
            
            #------------分类损失，文中没有清楚地说，只计算logits1的分类损失,还是logits1和logits2的分类损失
            logits = torch.cat([logits1[:, :args.base_class],logits2[:, :args.base_class]],dim=0)
            train_label_CE = torch.cat([train_label, train_label], dim=0)
            loss_CE = F.cross_entropy(logits, train_label_CE) #include softmax
            
            #---mask损失
            tl_cls.add(loss_CE.item(),logits.size(0))
            if args.debug3_maskLoss: #-----------------------debug3_maskLoss_9
                normal_id = trainloader.dataset.base_cls.index('normal')
                k_normal_ids = torch.where(train_label==normal_id)[0]
                if len(k_normal_ids) !=0:
                    normal_mask = mask_sigmoid1[k_normal_ids]
#                     normal_mask = torch.cat([mask_sigmoid1[k_normal_ids],mask_sigmoid2[k_normal_ids]], dim=0)
                    loss_mask = F.binary_cross_entropy(normal_mask,torch.zeros_like(normal_mask).detach())
                    loss_CE = loss_CE + args.mask_weight*loss_mask
                    tl_mask.add(loss_mask.item(),len(k_normal_ids))     
            #---mask损失
            
            loss = loss_CE + args.lambda_1*loss_CT
    
            tl.add(loss.item(),logits.size(0))
            tl_con.add(args.lambda_1*loss_CT.item(),logits.size(0))
            lgt=torch.cat([lgt,logits.cpu()])
            lbs=torch.cat([lbs,train_label_CE.cpu()])
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tl = tl.item()
        tl_cls = tl_cls.item()
        tl_con = tl_con.item()
        if args.procedure in ['singleLabel1']:
            ta = count_acc(lgt, lbs)
            ta_cls,_ = count_acc_perClass(lgt, lbs)
            if args.debug3_maskLoss:#-----------------------debug3_maskLoss_11
                tl_mask = tl_mask.item()
                print('--------epo {}, train, loss={:.4f} loss_cls={:.4f} loss_CT={:.4f} tl_mask={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch, tl,tl_cls,tl_con,tl_mask,ta,ta_cls))
            else:
                print('--------epo {}, train, loss={:.4f} loss_cls={:.4f} loss_CT={:.4f} acc_perSample={:.4f} acc_perClass={:.4f}'.format(epoch, tl,tl_cls,tl_con,ta,ta_cls))
        return tl, ta
  
 #  和01base一样  
    def test(self,model, testloader, epoch,args, session,validation=True, mode_test='test',result_list=None):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                _, logits,_ = model(data)#-----------------------debug3_maskLoss_12
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
    def replace_base_fc(self,trainset, transform, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
    
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base,
                                                  num_workers=args.num_workers, pin_memory=True, shuffle=False)
        trainloader.dataset.transform = transform
        trainloader.dataset.train = False ############Different from 01base
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                embedding = model(data)
                embedding = F.normalize(embedding, p=2, dim=-1)#-----------------------Different from 01base
    
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
    
        proto_list = []
    
        for class_index in range(args.base_class):
            data_index = torch.nonzero(label_list == class_index,as_tuple=False)
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
    
        proto_list = torch.stack(proto_list, dim=0)
    
        model.module.fc.weight.data[:args.base_class] = proto_list
    
        return model

