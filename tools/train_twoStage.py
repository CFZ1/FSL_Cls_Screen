import argparse
import importlib
from utils import *
import yaml

MODEL_DIR=None
DATA_DIR = 'data/'
PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    
    parser.add_argument('-project', type=str, default=PROJECT,
                        choices=['01fact','01base','01transformer_update2','01cec','01joint_training','01_FSCF','01base_episode','01ours2stage'])
    parser.add_argument('-default_arg_path', type=str, default=None)
    parser.add_argument('-dataset', type=str, default='mobilescreen_singleLabel',
                        choices=['mobilescreen_singleLabel','NEU','Magnetic'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    #------------------------ manifold mixup----------------
    parser.add_argument('-mixup',action='store_true',default=True,help='feature augment') #currently only support joint_training
    parser.add_argument('-alpha', type=float, default=0.5) #match with mixup
    parser.add_argument('-mixup_weight', type=float, default=0.5)
    parser.add_argument('-filterSame',action='store_true',default=False)
    #------------------------ manifold mixup----------------
       
    #-----------------only for cRT------
    parser.add_argument('-task_id', type=int, default=1)
    #-----------------only for cRT------

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-lr_newFc', type=float, default=0.1) #---------- only for 02finetune
    parser.add_argument('-optim', type=str, default='sgd', choices=['sgd','adam','mmcv_sgd']) #currently only support joint_training
    parser.add_argument('-bias_decay_mult', type=float, default=0.0)
    parser.add_argument('-norm_decay_mult', type=float, default=0.0)
    #-------------------lr-----------------
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone','Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    #-------------------lr-----------------
    
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', default=False,help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    #-----------------only for mobilescreen_singleLabel mobilescreen------
    parser.add_argument('-shot', type=int, default=2,help='novel class')
    #-----------------only for mobilescreen_singleLabel mobilescreen------
    
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_euclidean','ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_euclidean','avg_euclidean','ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    #-----------------only for fact------
    parser.add_argument('-balance', type=float, default=1.0)
    parser.add_argument('-loss_iter', type=int, default=200)
    parser.add_argument('-fact_alpha', type=float, default=0.5) #match with mixup
    parser.add_argument('-eta', type=float, default=0.1)
    #-----------------for fact------
    #------------------------ only for 03LPN_NLC----------------
    parser.add_argument('-rn', type=int, default=300,
                        help="graph construction types: "
                        "300: sigma is learned, alpha is fixed" +
                        "30:  both sigma and alpha learned") #300 or 30
    parser.add_argument('-k', type=int, default=20) #cannot exceed episode_way*(episode_shot+episode_query)
    parser.add_argument('-LPN_alpha', type=float, default=0.99) 
    #-----------------only for cec and 03LPN_NLC------
    # for episode learning
    parser.add_argument('-train_episode', type=int, default=100) #03LPN_NLC
    parser.add_argument('-episode_shot', type=int, default=5) #03LPN_NLC
    parser.add_argument('-episode_way', type=int, default=5) #03LPN_NLC
    parser.add_argument('-episode_query', type=int, default=15) #03LPN_NLC
    # for cec
    parser.add_argument('-lrg', type=float, default=0.1) #lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=15)
    parser.add_argument('-data_augment', type=str, default='normal',choices=['normal','01cec'])
    #-----------------for cec------

    

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    #-----------------only for cec------
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')
    #-----------------for cec------
    #-------------noly for gen_weight-----------
    parser.add_argument('-use_back',action='store_true', default=False)
    parser.add_argument('-use_label_embed',action='store_true', default=True)
    #-------------noly for gen_weight-----------
    
    #-------------noly for finetune-----------
    parser.add_argument('-reg_weight', type=float, default=0.1)
    #-------------noly for finetune-----------
    #-------------noly for 04transformer_update1-----------
    parser.add_argument('-debug1_relationAdd', action='store_true',default=False)
    parser.add_argument('-debug2_relationMask', action='store_true',default=False)
    parser.add_argument('-debug3_maskLoss', action='store_true',default=False)
    parser.add_argument('-mask_weight', type=float, default=0.1)
    parser.add_argument('-relation_mask', type=float, default=1.0)
    parser.add_argument('-mask_normalize', action='store_true',default=False)
    parser.add_argument('-save_data', action='store_true',default=False)
    parser.add_argument('-test_random', action='store_true',default=False)
    #-------------noly for 04transformer_update1-----------
    #-------------noly for 01_FSCF-----------
    parser.add_argument('-sigma', type=float, default=0.5)
    parser.add_argument('-lambda_1', type=float, default=0.5)
    #-------------noly for 01_FSCF-----------
    parser.add_argument('-test_runs', type=int, default=1000) ##04transformer_update1, multiple tests, calculate average results
    parser.add_argument('-check_test_result_inProcess', action='store_true',default=False)
    #-------------noly for 01ours2stage-----------
    parser.add_argument('-trainMode', type=int, default=1,choices=[1,2,3,4])
    parser.add_argument('-trainModeStage', type=int, default=2,choices=[1,2])
    #-------------noly for 01ours2stage-----------

    # about training
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true',default=False)
    parser.add_argument('-pth_workdir', type=str,default=None)
    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    #----------yaml highest level
    if args.default_arg_path is not None:
        with open(args.default_arg_path, 'r') as f:
            default_args = yaml.safe_load(f)
        parser.set_defaults(**default_args)
        args = parser.parse_args()
    #-----------------local debug，not a server，due to limited graphics memory, etc
    if args.debug:
        if 'mobilescreen' in args.dataset:
            args.dataroot = '/media/XXX/Elements/dataset/mobile_screen/0_few_shot'
        elif 'NEU' in args.dataset:
            args.dataroot = '/media/XXX/Elements/dataset/NEU-DET'
        elif 'Magnetic' in args.dataset:
            args.dataroot = '/media/XXX/Elements/dataset/Surface-Defect-Detection-master/Magnetic-Tile-Defect/'
        args.batch_size_base = 2
        args.test_batch_size = 2
        args.epochs_base = 1
        args.test_runs = 2
        args.gpu = '0'
        args.train_episode = 1
        args.episode_shot = 2
        args.episode_query = 1
        # args.model_dir = None
        args.low_shot = min(args.low_shot,args.episode_shot)
    #------------------------    
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    if args.trainModeStage == 1:
        trainer = importlib.import_module('models.%s.fscil_trainer_pretrain' % (args.project)).FSCILTrainer(args)
    else:
        trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()
