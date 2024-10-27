import torch
import matplotlib.pyplot as plt
import metrics 
from tqdm import tqdm

def evaluate_model_ch_wts_contrastive(datatype,ep,opts,model,loader,device,criterion,return_features=False):
    model.eval()
    true_labels = []
    logit_list = []
    states = []
    class_wise_count = {}
    class_wise_count = {}
    class_wise_correct = {}
    loss_meter = metrics.AverageMeter()

    with torch.no_grad():
        for i,sample in tqdm(enumerate(loader),total=len(loader)):
            motion_rep = sample['motion_rep'].type(torch.float).to(device)                
            input_to_net = motion_rep
            if opts.dataset in ['jhmdb','hmdb','le2i']:
                yt = sample['label'].type(torch.long).to(device)


            op,state,_ = model(input_to_net,sample)
            class_loss = criterion(op.squeeze(0), yt)
            logit_list += list(op)
            loss_meter.update(class_loss,len(sample['label']))

            if state is not None:
                states += list(state)
            true_labels+=list(yt)
            _, predicted = torch.max(op, 1)
            if opts.dataset != 'charades':
                for b_el in range(len(sample['label'])):
                    y_el = yt[b_el].cpu().numpy().item()
                    pred_el = predicted[b_el].cpu().numpy().item()
                    key_to_check = str(y_el)
                    is_correct = (y_el == pred_el)
                    #import ipdb; ipdb.set_trace()
                    if key_to_check in class_wise_count:
                        class_wise_count[key_to_check]+=1
                    else:
                        class_wise_count[key_to_check]=1
                    if is_correct:
                        if key_to_check in class_wise_correct:
                            class_wise_correct[key_to_check]+=1
                        else:
                            class_wise_correct[key_to_check]=1

    logit_list = torch.stack(logit_list)
    true_labels = torch.stack(true_labels)
    
    #breakpoint()
    metrics_to_return = {
        'accuracy':metrics.accuracy(logit_list,true_labels)[0].cpu().numpy()[0],
        'loss':loss_meter.avg
    }


    if opts.dataset in ['jhmdb','hmdb','le2i']:
        print('{} Epoch {} : {:05.3f}%'.format(datatype,ep,metrics_to_return['accuracy']))
        metrics_to_return['map'] = 0.0
        mean_class_wise_accuracy = 0
        num_classes = len(class_wise_count.keys())
        for key in class_wise_count:
            if key in class_wise_correct:
                mean_class_wise_accuracy += (class_wise_correct[key]*100/class_wise_count[key])
        mean_class_wise_accuracy/=num_classes
        metrics_to_return['m_class_wise_accuracy'] = mean_class_wise_accuracy

    return metrics_to_return
