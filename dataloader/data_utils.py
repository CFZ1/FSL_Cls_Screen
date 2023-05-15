import numpy as np
import torch
from dataloader.sampler import CategoriesSampler

def set_up_datasets(args):
    args.model = None
    if args.dataset == 'mobilescreen_singleLabel': #------------filter multi-label
        print('The final data set is: mobilescreen_singleLabel')
        import dataloader.mobile_screen.mobilescreen_singleLabel_v2_256 as Dataset
    if args.dataset == 'mobilescreen_singleLabel_FSCF': #------------filter multi-label
        print('The final data set is: mobilescreen_singleLabel_FSCF')
        import dataloader.mobile_screen.mobilescreen_singleLabel_v2_256_FSCF as Dataset 
    if args.dataset == 'NEU':
        print('The final data set is: NEU')
        import dataloader.NEU.NEU as Dataset      
    if args.dataset == 'NEU_FSCF':
        print('The final data set is: NEU_FSCF')
        import dataloader.NEU.NEU_FSCF as Dataset
    if args.dataset == 'Magnetic':  
        print('The final data set is: Magnetic')
        import dataloader.Magnetic.Magnetic as Dataset 
    if args.dataset == 'Magnetic_FSCF':  
        print('The final data set is: Magnetic_FSCF')
        import dataloader.Magnetic.Magnetic_FSCF as Dataset 
    if ('NEU' in args.dataset):
        if args.debug3_maskLoss == True:
            print('Error!!! NEU does not have normal images. We set debug3_maskLoss = False.')
            args.debug3_maskLoss = False  
    if ('mobilescreen' in args.dataset) or ('NEU' in args.dataset) or ('Magnetic' in args.dataset):
        args.base_class = len(Dataset.base_cls)
        args.num_classes= len(Dataset.base_cls+Dataset.novel_cls)
        args.way = len(Dataset.novel_cls)
        args.shot = args.shot
        args.sessions = 2
        args.procedure = 'singleLabel1'
        if args.train_episode ==-1:
            args.train_episode = int((50*30)/((args.episode_shot+25)*args.episode_way/5))
            print('args.train_episode: ',args.train_episode)

        if args.episode_way > args.base_class:
            print('change the value of episode_way: args.episode_way=args.base_class, %d'%args.episode_way,'---->%d'%args.base_class)
            args.episode_way =args.base_class 

    if args.dataset == 'mobilescreen_singleLabel_nway': #------------filter multi-label
        print('The final data set is: mobilescreen_singleLabel_nway')
        import dataloader.mobile_screen.mobilescreen_singleLabel_nway as Dataset
        args.base_class = len(Dataset.base_cls)
        args.num_classes= len(Dataset.base_cls+Dataset.novel_cls)
        args.way = len(Dataset.novel_cls)
        args.shot = args.shot
        args.sessions = 2
        args.procedure = 'singleLabel1'
    if args.dataset == 'mobilescreen': #------------ singleLabel + multi-label
        import dataloader.mobile_screen.mobilescreen as Dataset
        args.base_class = 3
        args.num_classes=4
        args.way = 1
        args.shot = args.shot
        args.sessions = 2
        if args.debug:
            args.imputSize = 200
        elif args.project =='03LPN_NLC':
            args.imputSize = 512            
        else:
            args.imputSize = 1024

        args.procedure = 'multiLabel'
        # args.model = 'resnet20'
    if args.dataset == 'coco': #------------ singleLabel + multi-label
        import dataloader.coco2017.coco as Dataset
        args.base_class = len(Dataset.base_cls)
        args.num_classes= len(Dataset.base_cls+Dataset.novel_cls)
        args.way = len(Dataset.novel_cls)
        args.shot = args.shot
        args.sessions = 2
        args.imputSize = 224
        args.procedure = 'multiLabel'
        print('使用coco数据集, 后续流程与mobilescreen一样')
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset =="manyshotcifar":
        import dataloader.cifar100.manyshot_cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    
    if args.dataset == 'manyshotcub':
        import dataloader.cub200.manyshot_cub as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = args.shot_num
        args.sessions = 11

    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'mini_imagenet_withpath':
        import dataloader.miniimagenet.miniimagenet_with_img as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    
    
    if args.dataset == 'manyshotmini':
        import dataloader.miniimagenet.manyshot_mini as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    
    if args.dataset == 'imagenet100':
        import dataloader.imagenet100.ImageNet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'imagenet1000':
        import dataloader.imagenet1000.ImageNet as Dataset
        args.base_class = 600
        args.num_classes=1000
        args.way = 50
        args.shot = 5
        args.sessions = 9

    args.Dataset=Dataset
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.ImageNet(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader



def get_base_dataloader_meta(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index)


    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        testset = args.Dataset.ImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list
