import torch
import numpy as np
import torch.optim as optim
import os

from torch.utils.data import DataLoader
from opts import parse_opts
import json
import pdb
from dataloaders import *
from models import get_model
import torch.nn as nn
from evaluation import *
import trainers
import itertools
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


opts = parse_opts()
extra_args = {}
if opts.tags != "":
    extra_args['tags'] = opts.tags
if opts.name is not None:
    extra_args['name'] = opts.name

set_random_seeds(opts.random_seed)
collate_fn_train = None
collate_fn_test = None


def split_dataset(samples_size, ratio=[7, 1, 2]):
    samples = list(range(samples_size))
    random.shuffle(samples)  # 随机打乱样本顺序

    # 计算每段的样本数
    total_ratio = sum(ratio)
    split_points = [int(samples_size * sum(ratio[:i + 1]) / total_ratio) for i in range(len(ratio))]

    # 按照比例分割样本
    split_samples = [samples[split_points[i - 1] if i > 0 else 0: split_points[i]] for i in range(len(ratio))]

    return split_samples


if opts.dataset == 'le2i':
    samples_size = 3571  # 这里硬编码，样本总数
    train_samples, val_samples, test_samples = split_dataset(samples_size, ratio=[7, 1, 2])
    print(f'train, val, test={len(train_samples), len(val_samples), len(test_samples)}')
    trainDataset = Le2i(sample_list=train_samples, pose_encoding_path='data/Fallnpy', label_path='data/le2i_labels.npy', opts=opts, transform_type='train')
    valDataset = Le2i(sample_list=val_samples, pose_encoding_path='data/Fallnpy/', label_path='data/le2i_labels.npy', opts=opts, transform_type='val')
    testDataset = Le2i(sample_list=test_samples, pose_encoding_path='data/Fallnpy/', label_path='data/le2i_labels.npy', opts=opts, transform_type='val')
    valDataset_whole = None
    opts.number_of_classes = 2

    ########################统计一下样本
#     count0, count1 = 0, 0
#     print(train_samples[:4])  # [714, 1282, 1423, 1735]

#     print('train_num:', len(train_samples))
#     print('val_num:', len(val_samples))

#     for i in range(len(train_samples)):
#         if trainDataset[i]['label'] == 1:
#             count1 += 1
#         else:
#             count0 += 1
#     print('train ---->','fall:', count1, 'notfall:', count0)

#     count0, count1 = 0, 0
#     for i in range(len(val_samples)):
#         if valDataset[i]['label'] == 1:
#             count1 += 1
#         else:
#             count0 += 1
#     print('val   ---->','fall:', count1, 'notfall:', count0)
########################统计一下样本

elif (opts.dataset == 'jhmdb'):
    path = 'metadata/JHMDB/'
    trainDataset = JHMDB(data_loc=path, pose_encoding_path='data/JHMDB/', file_name='jhmdb_train' + opts.train_split, opts=opts, transform_type='train')
    valDataset = JHMDB(data_loc=path, pose_encoding_path='data/JHMDB/', file_name='jhmdb_test' + opts.train_split, opts=opts, transform_type='val')
    valDataset_whole = None
    opts.number_of_classes = 21
elif (opts.dataset == 'hmdb'):
    path = 'metadata/HMDB51/'
    trainDataset = HMDB(data_loc=path, pose_encoding_path='data/HMDB51/', file_name='hmdb_train' + opts.train_split, opts=opts, transform_type='train')
    valDataset = HMDB(data_loc=path, pose_encoding_path='data/HMDB51/', file_name='hmdb_test' + opts.train_split, opts=opts, transform_type='val')
    valDataset_whole = HMDB(data_loc=path, pose_encoding_path='data/HMDB51/', file_name='hmdb_test' + opts.train_split, opts=opts, transform_type='val', get_whole_video=True)
    opts.number_of_classes = 51
else:
    raise NotImplementedError

# print("{:<20} {:<15}".format('Key','Value'))
# for k, v in vars(opts).items():
#     if(v is None):
#         continue
#     print("{:<20} {:<15}".format(k, v))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using {}'.format(device))

experiment_name = opts.name

trainLoader = DataLoader(trainDataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_workers, drop_last=False, pin_memory=True, collate_fn=collate_fn_train)
valLoader = DataLoader(valDataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.n_workers, pin_memory=True, collate_fn=collate_fn_test)
testLoader = DataLoader(testDataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.n_workers, pin_memory=True, collate_fn=collate_fn_test)
valLoader_whole = None
if valDataset_whole is not None:
    valLoader_whole = DataLoader(valDataset_whole, batch_size=1, shuffle=False, num_workers=0)

if not os.path.exists(opts.dataset):
    os.makedirs(opts.dataset)
if not os.path.exists(os.path.join(opts.dataset, experiment_name)):
    os.makedirs(os.path.join(opts.dataset, experiment_name))

with open(os.path.join(opts.dataset, opts.name, 'commandline_args.txt'), 'w') as f:
    json.dump(opts.__dict__, f, indent=2)

model = get_model(opts, device)
model = nn.DataParallel(model)
model.to(device)

if (opts.optimizer == 'adam'):
    optimizer = optim.Adam(model.parameters(), opts.learning_rate)
elif opts.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=opts.learning_rate, momentum=opts.momentum, weight_decay=opts.weight_decay)

if opts.scheduler == 'on_plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opts.lr_patience, verbose=True, factor=0.1)
elif opts.scheduler == 'MultiStepLR':
    lr_schedule = (opts.lr_schedule).split(',')
    lr_schedule = [int(stp) for stp in lr_schedule]
    cprint('Using LR schedule {}'.format(lr_schedule), 'yellow')
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule)

temperature_schedule = None

if opts.trainer_type == 'ch_wt_contrastive':
    trainers.default_trainer_ch_wts_contrastive(opts, model, valLoader, trainLoader, device, optimizer, temperature_schedule, scheduler, testLoader)
