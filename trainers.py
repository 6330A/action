from evaluation import *
from tqdm import tqdm
import torch
from termcolor import cprint
from utils import get_lr
from models import save_all_information
import torch.nn as nn
from contrastive import SupConLoss


def default_trainer_ch_wts_contrastive(opts, model, valLoader, trainLoader, device, optimizer, temperature_schedule, scheduler, testLoader):
    train_loss = []
    val_loss = []
    val_accuracy = []
    train_accuracy = []
    train_map = []
    val_map = []
    epoch_start = -1

    if opts.dataset in ['jhmdb', 'hmdb', 'le2i']:
        criterion = nn.CrossEntropyLoss()

    joint_names = valLoader.dataset.joint_names()
    if opts.use_background == 'False':
        joint_names.remove('BKG')
    start_contrastive, increase_for_contrastive = int(opts.contrastive_start), int(opts.contrastive_linear)

    sc_loss = SupConLoss(temperature=opts.temperature)
    with tqdm(total=opts.n_epochs, initial=epoch_start + 1) as t:
        for epoch in range(epoch_start + 1, opts.n_epochs):
            running_loss = 0.0
            if (epoch % opts.save_every == 0):
                val_metrics = evaluate_model_ch_wts_contrastive('val', epoch, opts, model, valLoader, device, criterion)
                # if opts.evaluate_on_train == 'True':
                #     train_metrics = evaluate_model_ch_wts_contrastive('train', epoch, opts,model,trainLoader,device,criterion)
                # else:                
                vl_loss, vl_acc, vl_map = val_metrics['loss'], val_metrics['accuracy'], val_metrics['map']

                if opts.scheduler == 'on_plateau':
                    if epoch > 0:
                        scheduler.step(vl_loss)

                val_accuracy.append(vl_acc)

                val_loss.append(vl_loss)
                val_map.append(vl_map)
                get_current_lr = get_lr(optimizer)

                is_best = save_all_information(opts.dataset + '/' + opts.name, epoch, train_accuracy, val_accuracy, train_loss, val_loss, model)

            model.train()

            for i, sample_train in enumerate(trainLoader):

                motion_rep = sample_train['motion_rep'].type(torch.float).to(device)
                motion_rep2 = sample_train['motion_rep_augmented'].type(torch.float).to(device)
                input_to_net = torch.cat([motion_rep, motion_rep2], 0)
                if opts.dataset in ['jhmdb', 'hmdb', 'le2i']:
                    y = sample_train['label'].type(torch.long).to(device)

                optimizer.zero_grad()
                op, ch_wts, contrastive_features = model(input_to_net, sample_train)
                repeated_y = torch.cat([y, y], 0)
                contrastive_features = torch.stack(torch.split(contrastive_features, split_size_or_sections=input_to_net.shape[0] // 2, dim=0), 2)
                contrastive_loss, mask = sc_loss(contrastive_features.permute(1, 0, 2, 3), labels=y, contrastive_loss_type=opts.contrastive_loss_type, debug=True)
                if torch.randn(1) > 1.0:
                    no_positive = (mask.sum(1) == 0).sum()
                    one_positive = (mask.sum(1) < 2).sum()
                    print(f'{no_positive}/{mask.shape[0]} instances with no positives. {one_positive}/{mask.shape[0]} instances with less than two positives')

                contrastive_loss = contrastive_loss.mean()
                class_loss = criterion(op.squeeze(0), repeated_y)

                if epoch < start_contrastive:
                    l1_scale = 0.0
                else:
                    l1_scale = min((epoch - start_contrastive) / increase_for_contrastive, 1.0)

                loss = class_loss + opts.loss_weights_contrastive * l1_scale * contrastive_loss

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if opts.scheduler == 'MultiStepLR':
                scheduler.step()

            get_current_lr = get_lr(optimizer)
            t.set_postfix(total_loss='{:05.3f}'.format(running_loss / len(trainLoader)), max_val_acc='{:05.3f}'.format(max(val_accuracy)), max_map='{:05.3f}'.format(max(val_map)),
                          lr='{:05.3f}'.format(get_current_lr), l1_scale='{:05.3f}'.format(l1_scale))
            t.update()

    val_metrics = evaluate_model_ch_wts_contrastive('val', epoch, opts, model, valLoader, device, criterion)
    vl_loss, vl_acc, vl_map = val_metrics['loss'], val_metrics['accuracy'], val_metrics['map']
    val_accuracy.append(vl_acc)
    val_loss.append(vl_loss)
    val_map.append(vl_map)
    is_best = save_all_information(opts.dataset + '/' + opts.name, epoch, train_accuracy, val_accuracy, train_loss, val_loss, model)
    print('Max val accuracy is {}'.format(max(val_accuracy)))
    print('Max val mAP is {}'.format(max(val_map)))

    print('Test .....','#'*50)
    model.load_state_dict(torch.load(opts.dataset + '/' + opts.name + '/' + '/model_best.t7'))  # 加载最优模型进行测试
    
    if opts.dataset == 'le2i':
        test_metrics = evaluate_model_ch_wts_contrastive('test', epoch, opts, model, testLoader, device, criterion)
        test_acc = test_metrics['accuracy']
        print(f'Test accuracy is {test_acc}')
    else:

        test_metrics = evaluate_model_ch_wts_contrastive('test', epoch, opts, model, valLoader, device, criterion)
        test_acc = test_metrics['accuracy']
        print(f'Test accuracy is {test_acc}')

