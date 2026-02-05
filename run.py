import argparse
import os
import torch
from exp.exp_supervised import Exp_Supervised
from exp.exp_finetune import Exp_Finetune
from exp.exp_pretrain.exp_pretrain_lead import Exp_Pretrain_LEAD
import random
import numpy as np
from utils.tools import compute_avg_std

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LEAD')

    # basic config
    parser.add_argument('--method', type=str, required=True, default='LEAD',
                        help='Overall method (combinations of task_name, model, model_id) name, '
                             'options: [LEAD, MOCO, Transformer, TCN]')
    parser.add_argument('--task_name', type=str, required=True, default='supervised',
                        help='task name, options:[supervised, pretrain_lead, pretrain_moco, finetune]')
    parser.add_argument('--model', type=str, required=True, default='LEAD',
                        help='backbone model name, options: [Transformer, TCN, LEAD]')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='Single-Dataset', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of all dataset folders')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument("--pretraining_datasets", type=str,
                        default="TDBRAIN",
                        help="List of datasets folder names for pretraining (No overlapping with downstream datasets).")
    parser.add_argument("--training_datasets", type=str,
                        default="ADFTD",
                        help="List of datasets folder names for linear probe, supervised, and finetune training.")
    parser.add_argument("--testing_datasets", type=str,
                        default="ADFTD",
                        help="List of datasets folder names for linear probe, supervised, and finetune validation and test.")
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints/LEAD/pretrain_lead/LEAD/P-12/',
                        help='location of pre-trained model checkpoints')
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
    parser.add_argument('--patch_len', type=int, default=50, help='patch_len used in PatchTST, BIOT,LEADv2')
    parser.add_argument('--stride', type=int, default=50, help='stride used in PatchTST, LEADv2')
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
                        help="a comma-seperated list of augmentation types (none, jitter or scale). "
                             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",)
    parser.add_argument("--no_inter_attn", action="store_true",
                        help="whether to use inter-attention in encoder, "
                             "using this argument means not using inter-attention", default=False)
    parser.add_argument("--no_temporal_block", action="store_true",
                        help="whether to use temporal block in encoder", default=False)
    parser.add_argument("--no_channel_block", action="store_true",
                        help="whether to use channel block in encoder", default=False)

    # LEAD/LEADv2 params
    parser.add_argument('--cross_patch_len', type=int, default=4, help='cross channel patch length used in LEAD')
    parser.add_argument('--scaled_channel_num', type=int, default=76, help='scaled channel number used in LEAD')
    parser.add_argument('--group_shuffle', action='store_true', help='use index group shuffle', default=False)
    parser.add_argument('--group_size', type=int, default=2, help='group size for group shuffle')
    parser.add_argument('--sampling_rate_list', type=str, default="200", help="list of all sampling rate")
    parser.add_argument('--use_subject_loss', action="store_true", help="use subject-regularized loss", default=False)
    parser.add_argument('--lambda2', type=float, default=0.75, help='weight of subject-level contrast, range in [0,1]')
    parser.add_argument('--channel_names', type=str, help='channel names', default='Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2')
    parser.add_argument('--montage_name', type=str, default='standard_1005', help='montage name for EEG channels, same as MNE library')
    parser.add_argument('--use_subject_vote', action='store_true', help='sample voting for subject-level performance', default=False)




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
        print("Supervised learning")
        Exp = Exp_Supervised
    elif args.task_name == 'pretrain_lead':
        print("Pretraining with LEAD method")
        Exp = Exp_Pretrain_LEAD
    elif args.task_name == 'finetune':
        print("Finetune")
        Exp = Exp_Finetune
    else:
        raise ValueError('task_name unknown, should be supervised, pretrain_lead, or finetune.')

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

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            (sample_val_metrics_dict, subject_val_metrics_dict,
             sample_test_metrics_dict, subject_test_metrics_dict, total_params) = exp.test(setting)
            total_params = total_params
            sample_val_metrics_dict_list.append(sample_val_metrics_dict)
            subject_val_metrics_dict_list.append(subject_val_metrics_dict)
            sample_test_metrics_dict_list.append(sample_test_metrics_dict)
            subject_test_metrics_dict_list.append(subject_test_metrics_dict)
            torch.cuda.empty_cache()
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
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            (sample_val_metrics_dict, subject_val_metrics_dict,
             sample_test_metrics_dict, subject_test_metrics_dict, total_params) = exp.test(setting, test=1)
            total_params = total_params
            sample_val_metrics_dict_list.append(sample_val_metrics_dict)
            subject_val_metrics_dict_list.append(subject_val_metrics_dict)
            sample_test_metrics_dict_list.append(sample_test_metrics_dict)
            subject_test_metrics_dict_list.append(subject_test_metrics_dict)
            torch.cuda.empty_cache()
        compute_avg_std(args, sample_val_metrics_dict_list, subject_val_metrics_dict_list,
                        sample_test_metrics_dict_list, subject_test_metrics_dict_list, total_params)

    else:
        raise ValueError('is_training should be 1 or 0, representing training or testing.')
