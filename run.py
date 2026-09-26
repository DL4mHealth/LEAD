import argparse
import os
import torch
from exp.exp_supervised import Exp_Supervised
from exp.exp_finetune import Exp_Finetune
from exp.exp_pretrain import Exp_Pretrain
from exp.exp_probe import Exp_Probe
import random
import numpy as np
from utils.tools import compute_avg_std

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LEAD')

    # basic config
    parser.add_argument('--method', type=str, required=True, default='LEAD',
                        help='Overall method name, e.g., LEAD')
    parser.add_argument('--task_name', type=str, required=True, default='supervised',
                        help='task name, options:[supervised, pretrain, finetune, probe]')
    parser.add_argument('--model', type=str, required=True, default='LEAD',
                        help='backbone model name; LEAD is required for contrastive pretraining')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='MultiDatasets', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of all dataset folders')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument("--pretraining_datasets", type=str, default="TDBrain",
                        help="Comma-separated dataset folder names for pretraining. This is the only dataset argument that may contain multiple datasets.")
    parser.add_argument("--training_dataset", type=str, default="ADFTD",
                        help="Single downstream dataset used for TRAIN/VAL/TEST, supervised learning, probing, and fine-tuning.")
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints/LEAD/pretrain/LEAD/',
                        help='pretrained checkpoint file or directory containing checkpoint.pth')
    parser.add_argument('--pretrain_init_path', type=str, default='',
                        help='optional checkpoint file/directory used to initialize contrastive pretraining')
    parser.add_argument('--classify_choice', type=str, default='multi_class',
                        help="classify AD vs HC, AD vs Non-AD (HC and all other classes, e.g, FTD), "
                             "HC vs Abnormal (All kinds of diseases) or multiclass, "
                             "options:[ad_vs_hc, ad_vs_nonad, hc_vs_abnormal, hc_vs_mci, ad_vs_mci, multi_class]")
    # parser.add_argument('--sampling_rate', type=int, default=128, help='frequency sampling rate')
    parser.add_argument('--ratio_a', type=float, default=0.8, help='training subject ratio')
    parser.add_argument('--ratio_b', type=float, default=0.9, help='training & validation subject ratio')
    parser.add_argument('--low_cut', type=float, default=0.5, help='low cut for bandpass filter')
    parser.add_argument('--high_cut', type=float, default=45, help='high cut for bandpass filter')



    # model define for baselines
    parser.add_argument('--top_k', type=int, default=1, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument("--freq", type=str, default="h", help="freq for time features encoding, "
                        "options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly],",)
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=50, help='patch_len used in PatchTST, BIOT,LEAD')
    parser.add_argument('--stride', type=int, default=50, help='stride used in PatchTST, LEAD')
    parser.add_argument('--resolution_list', type=str, default="2,4,6,8")
    parser.add_argument('--nodedim', type=int, default=10)
    parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[1, 1, 1, 1], help='num_blocks in each stage')
    parser.add_argument('--large_size', nargs='+', type=int, default=[31, 29, 27, 13], help='big kernel size')
    parser.add_argument('--small_size', nargs='+', type=int, default=[5, 5, 5, 5],
                        help='small kernel size for structral reparam')
    parser.add_argument('--dims', nargs='+', type=int, default=[128, 128, 128, 128], help='dmodels in each stage')
    parser.add_argument('--dw_dims', nargs='+', type=int, default=[256, 256, 256, 256],
                        help='dw dims in dw conv in each stage')
    parser.add_argument('--brain_regions', type=str, default='0,0,0,0,0,0,0,2,4,4,4,2,2,1,1,1,2,3,3',
                        help='brain region label for each channel used in CSBrain')
    # ADformer params
    parser.add_argument("--patch_len_list", type=str, default="4",
                        help="a list of patch len used in Medformer, ADformer")
    parser.add_argument("--up_dim_list", type=str, default="76",
                        help="a list of up dimension factor used in ADformer")
    parser.add_argument("--augmentations", type=str, default="flip,frequency,jitter,mask,channel,drop",
                        help="comma-separated augmentations: none, jitter, flip, frequency, mask, channel, patch, or drop. "
                             "Append a number to set strength, e.g. jitter0.1.",)
    parser.add_argument("--no_inter_attn", action="store_true",
                        help="whether to use inter-attention in encoder, "
                             "using this argument means not using inter-attention", default=False)
    parser.add_argument("--no_temporal_block", action="store_true",
                        help="whether to use temporal block in encoder", default=False)
    parser.add_argument("--no_channel_block", action="store_true",
                        help="whether to use channel block in encoder", default=False)

    # LEAD params
    parser.add_argument('--group_shuffle', action='store_true', default=False,
                        help='enable (subject_id, sampling_rate)-grouped pretraining batches')
    parser.add_argument('--group_size', type=int, default=2,
                        help='trials per (subject_id, sampling_rate) group; only active when group_shuffle is enabled')
    parser.add_argument('--pretrain_linear_probe', action='store_true', default=False,
                        help='enable downstream linear probing during/after pretraining (default: disabled)')
    parser.add_argument('--contrastive_token_ratio', type=float, default=1.0,
                        help='fraction of the CxP token grid kept by structured contrastive subsampling')
    parser.add_argument('--temporal_pos_type', type=str, default='fixed',
                        choices=['fixed', 'learnable'], help='fixed sinusoidal or learnable')
    parser.add_argument('--channel_pos_type', type=str, default='3D',
                        choices=['fixed', 'learnable', '3D'], help='fixed sinusoidal, learnable, or 3D electrode coordinates')
    parser.add_argument('--max_patch_positions', type=int, default=1024,
                        help='maximum temporal patch positions for LEAD learnable temporal positional embedding')
    parser.add_argument('--max_channel_positions', type=int, default=368,
                        help='maximum channel positions for LEAD learnable channel positional embedding (default: 368)')
    parser.add_argument('--use_sampling_embedding', action='store_true', default=False,
                        help='add learnable sampling-rate embedding in LEAD (default: disabled)')
    parser.add_argument('--sampling_rate_list', type=str, default="200",
                        help='sampling rates to load, e.g. 200 or 200,100,50; use "all" for all rates in y.dat')
    parser.add_argument('--use_subject_loss', action="store_true", help="use subject-regularized loss", default=False)
    parser.add_argument('--lambda2', type=float, default=0.75, help='weight of subject-level contrast, range in [0,1]')
    parser.add_argument('--use_subject_vote', action='store_true', help='sample voting for subject-level performance', default=False)
    parser.add_argument('--sample_per_subject', type=int, default=2500,
                        help='max sample number per subject for training, used in some datasets like TUEP')




    # optimization
    # parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument("--swa", action="store_true", help="use stochastic weight averaging", default=False)
    parser.add_argument('--no_normalize', action='store_true',
                        help='do not normalize data in data loader', default=False)
    parser.add_argument('--single_channel_mask', type=str, default='none',
                        help='channel mask for channel importance analysis, options: [None, Fp1, Fp2, ...]')
    # fixed: fixed split, mccv: monte carlo cross validation,
    # 5-fold: 5-fold cross validation, loso: leave-one-subject-out
    parser.add_argument('--cross_val', type=str, default='mccv',
                        help='cross validation methods, options: [fixed, mccv, 5-fold, loso]')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.task_name == 'supervised':
        print("Supervised learning from scratch")
        Exp = Exp_Supervised
    elif args.task_name == 'pretrain':
        print("Subject contrastive pretraining with structured token subsampling")
        Exp = Exp_Pretrain
    elif args.task_name == 'finetune':
        print("Downstream fine-tuning from checkpoint")
        Exp = Exp_Finetune
    elif args.task_name == 'probe':
        print("GPU linear probing with frozen LEAD backbone")
        Exp = Exp_Probe
    else:
        raise ValueError('task_name unknown, should be one of: supervised, pretrain, finetune, probe.')

    total_params = 0
    sample_val_metrics_dict_list = []
    subject_val_metrics_dict_list = []
    sample_test_metrics_dict_list = []
    subject_test_metrics_dict_list = []
    if args.is_training == 1:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            # setting record of experiments
            args.seed = seed
            setting = 'nh{}_el{}_dm{}_df{}_seed{}'.format(
                # args.model_id,
                # args.features,
                # args.seq_len,
                # args.label_len,
                # args.pred_len,
                args.n_heads,
                args.e_layers,
                # args.d_layers,
                args.d_model,
                args.d_ff,
                # args.factor,
                # args.embed,
                # args.distil,
                # args.des,
                args.seed
            )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if args.task_name == 'pretrain' and not args.pretrain_linear_probe:
                print('>>>>>>>pretraining linear probe disabled; skipping downstream evaluation<<<<<<<<<<<<<<<<')
                torch.cuda.empty_cache()
                continue

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            (sample_val_metrics_dict, subject_val_metrics_dict,
             sample_test_metrics_dict, subject_test_metrics_dict, total_params) = exp.test(setting)
            total_params = total_params
            sample_val_metrics_dict_list.append(sample_val_metrics_dict)
            subject_val_metrics_dict_list.append(subject_val_metrics_dict)
            sample_test_metrics_dict_list.append(sample_test_metrics_dict)
            subject_test_metrics_dict_list.append(subject_test_metrics_dict)
            torch.cuda.empty_cache()

        if args.task_name == 'pretrain' and not args.pretrain_linear_probe:
            print('Pretraining completed without linear probing. Add --pretrain_linear_probe to enable it.')
        else:
            compute_avg_std(args, sample_val_metrics_dict_list, subject_val_metrics_dict_list,
                            sample_test_metrics_dict_list, subject_test_metrics_dict_list, total_params)

    elif args.is_training == 0:
        # seed_list = [67, 69, 64, 58, 44, 104, 105, 91, 103, 85, 107, 109, 119, 120, 118]
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            args.seed = seed
            setting = 'nh{}_el{}_dm{}_df{}_seed{}'.format(
                # args.model_id,
                # args.features,
                # args.seq_len,
                # args.label_len,
                # args.pred_len,
                args.n_heads,
                args.e_layers,
                # args.d_layers,
                args.d_model,
                args.d_ff,
                # args.factor,
                # args.embed,
                # args.distil,
                # args.des,
                args.seed
            )

            exp = Exp(args)  # set experiments
            if args.task_name == 'pretrain' and not args.pretrain_linear_probe:
                print('Pretraining linear probe is disabled. Add --pretrain_linear_probe to evaluate a pretraining checkpoint.')
                torch.cuda.empty_cache()
                continue

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            (sample_val_metrics_dict, subject_val_metrics_dict,
             sample_test_metrics_dict, subject_test_metrics_dict, total_params) = exp.test(setting, test=1)
            total_params = total_params
            sample_val_metrics_dict_list.append(sample_val_metrics_dict)
            subject_val_metrics_dict_list.append(subject_val_metrics_dict)
            sample_test_metrics_dict_list.append(sample_test_metrics_dict)
            subject_test_metrics_dict_list.append(subject_test_metrics_dict)
            torch.cuda.empty_cache()

        if args.task_name == 'pretrain' and not args.pretrain_linear_probe:
            print('No pretraining linear-probe evaluation was run.')
        else:
            compute_avg_std(args, sample_val_metrics_dict_list, subject_val_metrics_dict_list,
                            sample_test_metrics_dict_list, subject_test_metrics_dict_list, total_params)

    else:
        raise ValueError('is_training should be 1 or 0, representing training or testing.')
