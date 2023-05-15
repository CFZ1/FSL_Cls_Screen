import abc
import torch
import os
from dataloader.data_utils import set_up_datasets

from utils import (
    ensure_path,
    Averager, Timer
)
import time
import torch.nn as nn
from copy import deepcopy
import numpy as np
from terminaltables import AsciiTable
import random


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args,MYNET):
        self.args = args
        self.set_save_path()
        
        self.args = set_up_datasets(self.args)
        
        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
            if 'module.dummy_orthogonal_classifier.weight' in self.best_model_dict.keys(): #####230413
                self.best_model_dict['module.dummy_orthogonal_classifier.weight'] = self.best_model_dict['module.dummy_orthogonal_classifier.weight'][:self.args.num_classes-self.args.base_class,:]
                
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()
        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['avg_acc_val'] = 0.0
        self.trlog['avg_acc_test'] = 0.0
        self.trlog['max_acc'] = [0.0] * args.sessions
        
        
    def get_dataloader(self, session, base_meta = False):
        trainloader, valloader, testloader = None,None,None
        if self.args.procedure in ['multiLabel','singleLabel1']:
            if session == 0:
                if base_meta:
                    trainset, trainloader, valloader, testloader = self.args.Dataset.get_base_dataloader_meta(self.args, session)
                else:
                    trainset, trainloader, valloader, testloader = self.args.Dataset.get_base_dataloader(self.args, session)
            else:
                trainset = self.args.Dataset.get_new_dataloader_manyRuns(self.args, session)
        else:
            print('not implement')
        return trainset, trainloader, valloader, testloader
        
    def get_optimizer_base(self):

        if self.args.optim == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                        weight_decay=self.args.decay)
        elif self.args.optim == 'adam': 
            optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr_base, weight_decay=self.args.decay)
        elif self.args.optim == 'mmcv_sgd':
            from mmcv.runner import build_optimizer
            optimizer = build_optimizer(self.model,dict(type='SGD', lr=self.args.lr_base, momentum=0.9, weight_decay=self.args.decay,paramwise_cfg=dict(bias_decay_mult=self.args.bias_decay_mult,norm_decay_mult =self.args.norm_decay_mult)))
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler
    
    def set_save_path(self):
        self.args.save_path = os.path.join(self.args.dataroot, "0workdirs")
        if self.args.pth_workdir == None:
            self.args.save_path = os.path.join(self.args.save_path, self.args.project)
        else:
            self.args.save_path = os.path.join(self.args.save_path, self.args.pth_workdir)
        self.args.save_path = os.path.join(self.args.save_path,self.args.dataset+time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        ensure_path(self.args.save_path)
        return None


    @abc.abstractmethod
    def train(self):
        pass
    @abc.abstractmethod
    def base_train(self,model, trainloader, optimizer, scheduler, epoch, args, base_train_args):
        pass
    @abc.abstractmethod
    def test(self,model, testloader, epoch,args,session,validation,mode_test):
        pass
    
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
    
        proto_list = []
    
        for class_index in range(args.base_class):
            data_index = torch.nonzero(label_list == class_index,as_tuple=False)
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
    
        proto_list = torch.stack(proto_list, dim=0)
    
        model.module.fc.weight.data[:args.base_class] = proto_list
    
        return model
    def cons_mobilescreenV2(self, train_set, run_path=None): #####230413, mobilescreen_singleLabel_v2_256.py:novel_cls = ['floater', 'watermark']
        train_idx_list = []
        test_idx_list = []
        if run_path==None:
            novel_idx_lists = train_set.get_novel_ids()
            train_idx_list_can = []
            train_idx_list_can_flatten = []
            for novel_idx_list in novel_idx_lists:
                smaple_list = random.sample(novel_idx_list, 15)
                test_idx_list.extend(smaple_list)
                train_idx_list_can.append([i for i in novel_idx_list if i not in smaple_list])
            for novel_idx_list in train_idx_list_can:
                train_idx_list.extend(random.sample(novel_idx_list, self.args.shot))
                train_idx_list_can_flatten.extend(novel_idx_list)
            test_idx_list = [i for i in list(range(0,len(train_set))) if i not in train_idx_list_can_flatten]
        else:
            print('data_path')
            train_idx_list_can = [[] for x in range(len(train_set.novel_cls))]
            run_trainData = [x.strip() for x in open(run_path).readlines()]
            for idx,target_id in enumerate(train_set.data):
                if target_id.split(train_set.img_path+'/')[-1] in run_trainData: #self.args.shot
                    for ii,novel_cls_i in enumerate(train_set.novel_cls):
                        if novel_cls_i in target_id.split(train_set.img_path+'/')[-1]:
                            train_idx_list_can[ii].append(idx)
            for novel_idx_list in train_idx_list_can:
                train_idx_list.extend(random.sample(novel_idx_list, self.args.shot))
                
            run_testData = [x.strip() for x in open(run_path.split('.')[0]+'_test.txt').readlines()]
            for idx,target_id in enumerate(train_set.data):
                if target_id.split(train_set.img_path+'/')[-1] in run_testData:
                    test_idx_list.append(idx)
        #------------------------unchanged-------------------
        test_set = deepcopy(train_set)
        train_set_new = deepcopy(train_set)
        train_set_new.data = [train_set_new.data[i] for i in train_idx_list]
        train_set_new.targets = [train_set_new.targets[i] for i in train_idx_list]
        test_set.data = [test_set.data[i] for i in test_idx_list]
        test_set.targets = [test_set.targets[i] for i in test_idx_list]
        if self.args.batch_size_new == 0:
            batch_size_new = train_set_new.targets.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=train_set_new, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=self.args.num_workers, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(dataset=train_set_new, batch_size=self.args.batch_size_new, shuffle=True,
                                      num_workers=self.args.num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False,
                                 num_workers=self.args.num_workers, pin_memory=True) 
        return train_set_new, trainloader,testloader
    def cons_mobilescreen(self, train_set, run_path=None):
        train_idx_list = []
        if run_path==None:
            novel_idx_lists = train_set.get_novel_ids()
            for novel_idx_list in novel_idx_lists:
                train_idx_list.extend(random.sample(novel_idx_list, self.args.shot))
        else:
            print('data_path')
            run_trainData = [x.strip() for x in open(run_path).readlines()]
            for idx,target_id in enumerate(train_set.data):
                if target_id.split(train_set.img_path+'/')[-1] in run_trainData:
                    train_idx_list.append(idx)
                    
        test_idx_list = [i for i in list(range(0,len(train_set))) if i not in train_idx_list]
        test_set = deepcopy(train_set)
        train_set_new = deepcopy(train_set)
        train_set_new.data = [train_set_new.data[i] for i in train_idx_list]
        train_set_new.targets = [train_set_new.targets[i] for i in train_idx_list]
        test_set.data = [test_set.data[i] for i in test_idx_list]
        test_set.targets = [test_set.targets[i] for i in test_idx_list]
        # train_set_new.add_noise_ONoff = False #----------------------------230412--4
        # test_set.add_noise_ONoff = True
        if self.args.batch_size_new == 0:
            batch_size_new = train_set_new.targets.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=train_set_new, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=self.args.num_workers, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(dataset=train_set_new, batch_size=self.args.batch_size_new, shuffle=True,
                                      num_workers=self.args.num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False,
                                 num_workers=self.args.num_workers, pin_memory=True) 
        return train_set_new, trainloader,testloader
    def cons_NEU(self, train_set):
        train_set_new = deepcopy(train_set[0])
        test_set = deepcopy(train_set[1])
        novel_idx_lists = train_set_new.get_novel_ids()
        train_idx_list = []
        for novel_idx_list in novel_idx_lists:
            train_idx_list.extend(random.sample(novel_idx_list, self.args.shot))

        train_set_new.data = [train_set_new.data[i] for i in train_idx_list]
        train_set_new.targets = [train_set_new.targets[i] for i in train_idx_list]

        if self.args.batch_size_new == 0:
            batch_size_new = train_set_new.targets.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=train_set_new, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=self.args.num_workers, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(dataset=train_set_new, batch_size=self.args.batch_size_new, shuffle=True,
                                      num_workers=self.args.num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False,
                                 num_workers=self.args.num_workers, pin_memory=True) 
        return train_set_new, trainloader,testloader
        
    
    def train_s0(self, base_meta, result_list, base_train_args=None):
        '''
        self.base_train
        self.test
        self.replace_base_fc
        
        '''
        session = 0 
        #--------------1: 加载数据。 train：base类别（batch or meta)；val：base类别；test：base类别
        #################batch or meta?
        train_set, trainloader, valloader, testloader = self.get_dataloader(session,base_meta=base_meta)
        if self.args.project in ['01joint_training']:
            self.testloader = testloader

        self.model.load_state_dict(self.best_model_dict)
        if self.args.procedure in ['multiLabel','singleLabel1']:
            print("session: [%d]; " % session, "classes:", train_set.base_cls )
        else:
            print("session: [%d]; " % session, "classes:", np.unique(train_set.targets))
        #--------------2: 加载optimizer, scheduler 
        optimizer, scheduler = self.get_optimizer_base()

        for epoch in range(self.args.epochs_base):    
            #--------------3: base_train数据训练模型
            if self.args.project in ['01cec']:
                self.model.eval()
            tl, ta = self.base_train(self.model, trainloader, optimizer, scheduler, epoch, self.args, base_train_args)
            #--------------4: base_val数据测试模型: loss,acc_perSample,acc_perClass
            if self.args.project in ['01cec']:
                self.model = self.replace_base_fc(train_set, testloader.dataset.transform, self.model, self.args)
                self.model.module.mode = 'avg_cos'
            if self.args.project in ['01transformer_update2'] and (self.args.episode_way !=self.args.base_class):#------------------------1124
                self.model = self.replace_base_fc(train_set, valloader.dataset.transform, self.model, self.args)
                self.model.module.mode = self.args.base_mode#------------------------1124
            tsl, tsa, ta_cls_perCls = self.test(self.model, valloader, epoch, self.args, session,validation=False,mode_test='val')
            #--------------4: base_test数据测试模型: loss,acc_perSample,acc_perClass
            if self.args.check_test_result_inProcess:
                tsl_test, tsa_test, ta_cls_perCls_test = self.test(self.model, testloader, epoch, self.args, session,validation=False,mode_test='test')
            #--------------5:保存最佳模型，根据base_val的acc_perSample还是acc_perClass
            compare_metric = np.mean(ta_cls_perCls) #tsa 
            if (compare_metric * 100) >= self.trlog['max_acc'][session]:
                self.trlog['max_acc'][session] = float('%.3f' % (compare_metric * 100))
                self.trlog['max_acc_epoch'] = epoch
                save_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                torch.save(optimizer.state_dict(), os.path.join(self.args.save_path, 'optimizer_best.pth'))
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                               self.trlog['max_acc'][session]))

            self.trlog['train_loss'].append(tl)
            self.trlog['train_acc'].append(ta)
            self.trlog['test_loss'].append(tsl)
            self.trlog['test_acc'].append(compare_metric)
            lrc = scheduler.get_last_lr()[0]
            result_list.append(
                'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                    epoch, lrc, tl, ta, tsl, compare_metric))
            #--------------6:改变学习率
            scheduler.step()

        result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
            session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
        #--------------7:使用base_train数据, Replace the fc with average embedding
        if not self.args.not_data_init:
            self.model.load_state_dict(self.best_model_dict)
            self.model = self.replace_base_fc(train_set, valloader.dataset.transform, self.model, self.args)
            best_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_average.pth')
            print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
            self.best_model_dict = deepcopy(self.model.state_dict())
            torch.save(dict(params=self.model.state_dict()), best_model_dir)

            self.model.module.mode = self.args.base_mode
            #--------------7.1:base_val数据测试模型: loss,acc_perSample,acc_perClass
            tsl, tsa, ta_cls_perCls = self.test(self.model, valloader, 0, self.args, session,validation=False,mode_test='val')
            compare_metric = np.mean(ta_cls_perCls) #tsa 
            self.trlog['avg_acc_val'] = float('%.3f' % (compare_metric * 100))
            if (compare_metric * 100) >= self.trlog['max_acc'][session]:
                print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
            #--------------7.1:base_test数据测试模型: loss,acc_perSample,acc_perClass
            print('----------------results on test dataset--------------------------')
            tsl_test, tsa_test, ta_cls_perCls_test = self.test(self.model, testloader, 0, self.args, session,validation=False,mode_test='test')
            self.trlog['avg_acc_test'] = float('%.3f' % (np.mean(ta_cls_perCls_test) * 100))
    def train_s1_manyRuns(self, result_list):
        session = 1
        #--------------1: 加载数据。 train_set: base_test和novel所有, trainloader=valloader=testloader=None
        train_set, trainloader, valloader, testloader = self.get_dataloader(session)
        self.best_model_dict['module.fc.weight'] = self.best_model_dict['module.fc.weight'][:self.args.num_classes,:]###1031
        self.model.load_state_dict(self.best_model_dict)
        if self.args.procedure in ['multiLabel','singleLabel1']:
            if ('mobilescreen' in self.args.dataset) or ('Magnetic' in self.args.dataset):
                print("session: [%d]; " % session, "classes:", train_set.novel_cls )
            elif 'NEU' in self.args.dataset: 
                print("session: [%d]; " % session, "classes:", train_set[0].novel_cls )
        else:
            print("training session: [%d]" % session)
        ta_cls_perCls_list = []
        #--------------2: 多次结果求平均
        for run in range(self.args.test_runs):
            print('-----------------',run,'-----------------')###1031
            self.model.module.mode = self.args.new_mode
            self.model.eval()  
            #---------------------2.1:创建训练集trainloader和测试集testloader。训练集为n_novel-way，k-shot,其他数据都归测试集 ------------------
            # NEU和mobilescreen不同，NEU的novel类别已经事先划分为train+test了
            if ('mobilescreen' in self.args.dataset) or ('Magnetic' in self.args.dataset):
                if self.args.test_random:
                    run_path = None  
                else:
                    # run_path = os.path.join(self.args.dataroot,'split_'+str(self.args.way)+'w'+str(self.args.shot)+'s/r'+str(run+1)+'.txt')
                    run_path = os.path.join(self.args.dataroot,'split_'+str(2)+'w'+str(15)+'s/r'+str(run)+'.txt')
                    if not os.path.exists(run_path):
                        run_path = None                  
                # train_set_new, trainloader,testloader = self.cons_mobilescreen(train_set,run_path)
                train_set_new, trainloader,testloader = self.cons_mobilescreenV2(train_set,run_path) #####230413 
            elif 'NEU' in self.args.dataset:
                train_set_new, trainloader,testloader = self.cons_NEU(train_set)  
            #---------------------创建训练集和测试集------------------
            #---------------------2.2:trainloader更新模型
            self.model.module.update_fc(trainloader, np.unique(train_set_new.targets), session)
#             print(train_set_new.data) #------------230402
#             result_list.append(train_set_new.data) #------------230402
             #---------------------2.3:testloader测试模型: loss,acc_perSample,acc_perClass
            if self.args.project in ['01fact']:
                tsl, tsa, ta_cls_perCls = self.test_intergrate(self.model, testloader, 0,self.args, session,validation=False)
            else:
#                 self.model.module.protos_mask = torch.ones([self.args.num_classes,512])
                tsl, tsa, ta_cls_perCls = self.test(self.model, testloader, 0, self.args, session, validation=False, result_list = result_list)
            #------------save_data_1s
            # if 'mobilescreen' in self.args.dataset and (self.args.save_data):
            #     data_save = [train_set_new.data[i].split(train_set_new.img_path+'/')[-1] for i in range(len(train_set_new))] 
            #     path = os.path.join(self.args.dataroot,'split_'+str(self.args.way)+'w'+str(self.args.shot)+'s_save/r'+str(run)+'n'+str(np.mean(ta_cls_perCls[self.args.base_class:self.args.num_classes])*1000).split('.')[0]+'a'+str(np.mean(ta_cls_perCls)*1000).split('.')[0]+'.txt')
            #     str1 = '\n'
            #     f=open(path,"w")
            #     f.write(str1.join(data_save))
            #     f.close()
            if 'mobilescreen' in self.args.dataset and (self.args.save_data): #####230413
                train_save = [train_set_new.data[i].split(train_set_new.img_path+'/')[-1] for i in range(len(train_set_new))] 
                test_save = [testloader.dataset.data[i].split(testloader.dataset.img_path+'/')[-1] for i in range(len(testloader.dataset))]
                train_path = os.path.join(self.args.dataroot,'split_'+str(self.args.way)+'w'+str(self.args.shot)+'s_save/r'+str(run)+'.txt')
                test_path = train_path.split('.')[0]+'_test.txt'
                str1 = '\n'
                f=open(train_path,"w")
                f.write(str1.join(train_save))
                f.close()
                f=open(test_path,"w")
                f.write(str1.join(test_save))
                f.close()
            #------------save_data_1s
            ta_cls_perCls_list.append(ta_cls_perCls)
            compare_metric = np.mean(ta_cls_perCls) #tsa 
            self.trlog['max_acc'][session] = float('%.3f' % (compare_metric * 100))
            save_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=self.model.state_dict()), save_model_dir)
            if self.args.project in ['01transformer_update2']:
                if self.args.debug2_relationMask:
                    torch.save({'protos_mask':self.model.module.protos_mask}, os.path.join(self.args.save_path, 'session' + str(session) + '_protos_mask.pth'))
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('Saving model to :%s' % save_model_dir)
            print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

            result_list.append('run {}, Session {}, test Acc {:.3f}'.format(run,session, self.trlog['max_acc'][session]))
            if run ==9:
                temp = np.array(ta_cls_perCls_list).mean(0)
                print('前10----Session1 (base/novel/all): {:.3f}/{:.3f}/{:.3f}'.format(temp[:len(train_set_new.base_cls)].mean()*100,temp[len(train_set_new.base_cls):].mean()*100,temp.mean()*100))
                results_print = {}
                results_print["name"] = tuple(testloader.dataset.base_cls+testloader.dataset.novel_cls)
                results_print["result"] =tuple( ['{va_int:0.3f}%'.format(va_int=temp[ind]*100) for ind in range(temp.shape[0])] )
                table_data = [results_print["name"],results_print["result"]]
                table = AsciiTable(table_data)
                print('------------前10--------------') # ----------1103
                print(table.table)
                result_list.append('前10----Session1 (base/novel/all): {:.3f}/{:.3f}/{:.3f}'.format(temp[:len(train_set_new.base_cls)].mean()*100,temp[len(train_set_new.base_cls):].mean()*100,temp.mean()*100))
        #------------230402
#         save_tensor_pth = os.path.join(self.args.save_path, str(self.args.dataset)+'_'+str(self.args.way)+'w'+str(self.args.shot)+'s_'+'session' + str(session)+'_result.pth')
#         try:
#             torch.save(self.pred_reslut, save_tensor_pth)
#         except:
#             print('No prediction results need to be saved')
        #------------230402
        ta_cls_perCls_ = np.array(ta_cls_perCls_list)
        #------------save_data_2e
        if 'mobilescreen' in self.args.dataset and (self.args.save_data):
            save_tensor_pth = os.path.join(self.args.dataroot, 'split_'+str(self.args.way)+'w'+str(self.args.shot)+'s_save/result.pth')
            torch.save({'ta_cls_perCls_':torch.from_numpy(ta_cls_perCls_)}, save_tensor_pth)
        #------------save_data_2e
        test_runs_2 = 100 #!TODO----------1103
        ta_cls_perCls_ = ta_cls_perCls_[:test_runs_2,:].mean(0)#[100,num_cls]----------1103
        #--------------------打印100次结果的均值-----------
        results_print = {}
        results_print["name"] = tuple(testloader.dataset.base_cls+testloader.dataset.novel_cls)
        results_print["result"] =tuple( ['{va_int:0.3f}%'.format(va_int=ta_cls_perCls_[ind]*100) for ind in range(ta_cls_perCls_.shape[0])] )
        table_data = [results_print["name"],results_print["result"]]
        table = AsciiTable(table_data)
        print('-------------Mean value of %d runs-------------' % test_runs_2) # ----------1103
        print(table.table)
        result_list.append('-------------Mean value of %d runs-------------' % test_runs_2)
        result_list.append(table.table)
        #--------------------打印100次结果的均值-----------
        print('-------------summary-------------')
        result_list.append('-------------summary-------------')
        print("base classes:", train_set_new.base_cls)
        print("novel classes:", train_set_new.novel_cls)
        print('Session0_val: {:.3f} (bast_epoch: {})'.format(self.trlog['max_acc'][0],self.trlog['max_acc_epoch']))
        print('Session0_avg_val: {:.3f}'.format(self.trlog['avg_acc_val']))
        print('Session0_avg_test: {:.3f}'.format(self.trlog['avg_acc_test']))
        print('Session1 (base/novel/all): {:.3f}/{:.3f}/{:.3f}'.format(ta_cls_perCls_[:len(train_set_new.base_cls)].mean()*100,ta_cls_perCls_[len(train_set_new.base_cls):].mean()*100,ta_cls_perCls_.mean()*100))
        result_list.append('Session0_val: {:.3f} (bast_epoch: {})'.format(self.trlog['max_acc'][0],self.trlog['max_acc_epoch']))
        result_list.append('Session0_avg_val: {:.3f}'.format(self.trlog['avg_acc_val']))
        result_list.append('Session0_avg_test: {:.3f}'.format(self.trlog['avg_acc_test']))
        result_list.append('Session1 (base/novel/all): {:.3f}/{:.3f}/{:.3f}'.format(ta_cls_perCls_[:len(train_set_new.base_cls)].mean()*100,ta_cls_perCls_[len(train_set_new.base_cls):].mean()*100,ta_cls_perCls_.mean()*100))
        if self.args.test_runs == 1000:
            ta_cls_perCls_ = np.array(ta_cls_perCls_list) #[1000,num_classes]
            base_ta = ta_cls_perCls_[:,:self.args.base_class].mean(1) #base,[1000]
            novel_ta = ta_cls_perCls_[:,self.args.base_class:self.args.num_classes].mean(1) #novel,[1000]
            all_ta = ta_cls_perCls_.mean(1) #all,[1000]
            ta_cls_perCls_ = np.stack((base_ta,novel_ta,all_ta),axis=1)#[1000,3]
            #过滤掉内容重复
            ta_cls_perCls_ = np.unique(ta_cls_perCls_,axis=0)
            #先按照第2列从大到小排序，如果相同，再按照第3列大到小排序
            ta_cls_perCls_list = ta_cls_perCls_.tolist()
            ta_cls_perCls_list = sorted(ta_cls_perCls_list,key=lambda x: (-x[1],-x[2]))
            ta_cls_perCls_ = np.array(ta_cls_perCls_list)
            
            ta_cls_perCls_ = ta_cls_perCls_[:100,:]#[100,3]
            ta_cls_perCls_ = ta_cls_perCls_.mean(0)#[100,3]
            print('100 out of 1000 (base/novel/all): {:.3f}/{:.3f}/{:.3f}'.format(ta_cls_perCls_[0]*100,ta_cls_perCls_[1]*100,ta_cls_perCls_[2]*100)) 
        print('    ')
