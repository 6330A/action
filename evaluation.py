import torch
import metrics
from tqdm import tqdm
from matrix import Matrix

def evaluate_model_ch_wts_contrastive(datatype, ep, opts, model, loader, device, criterion, return_features=False):
    model.eval()
    class_nums = None
    labels = None
    if opts.dataset == 'hmdb':
        class_nums = 51
        labels = ['brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb', 'climb_stairs', 'dive', 'draw_sword', 'dribble', 'drink', 'eat', 'fall_floor', 'fencing', 'flic_flac', 'golf', 'handstand', 'hit', 'hug', 'jump', 'kick', 'kick_ball', 'kiss', 'laugh', 'pick', 'pour', 'pullup', 'punch', 'push', 'pushup', 'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault', 'stand', 'swing_baseball', 'sword', 'sword_exercise', 'talk', 'throw', 'turn', 'walk', 'wave']

    elif opts.dataset == 'jhmdb':
        class_nums = 21
        labels=['brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'stand', 'swing_baseball', 'throw', 'walk', 'wave']
    elif opts.dataset == 'le2i':
        class_nums = 2
        labels=['not fall', 'fall']

    matrix = Matrix(class_nums=class_nums, labels=labels)

    true_labels = []
    logit_list = []

    loss_meter = metrics.AverageMeter()

    with torch.no_grad():
        for i, sample in tqdm(enumerate(loader), total=len(loader)):
            motion_rep = sample['motion_rep'].type(torch.float).to(device)
            input_to_net = motion_rep
            yt = sample['label'].type(torch.long).to(device)

            op, _, _ = model(input_to_net, sample)
            class_loss = criterion(op.squeeze(0), yt)
            loss_meter.update(class_loss, len(sample['label']))
            print("看看 evaluation 中 len(sample['label'])  : ", len(sample['label']))

            logit_list += list(op)
            true_labels += list(yt)
            _, predicted = torch.max(op, 1)

            for b_el in range(len(sample['label'])):
                y_el = yt[b_el].cpu().numpy().item()
                pred_el = predicted[b_el].cpu().numpy().item()
                matrix.update(pred_el, y_el)

    logit_list = torch.stack(logit_list)
    true_labels = torch.stack(true_labels)

    # 计算top_k精度
    top_k = (1, ) if opts.dataset == 'le2i' else (1, 5)
    acc = metrics.accuracy(logit_list, true_labels, top_k)

    # 计算损失
    loss = loss_meter.avg

    # map写的是0.0,至于Mean Class-Wise Accuracy暂时不写,通过混淆矩阵matrix可得
    metrics_to_return = {
        'accuracy': acc[0].cpu().numpy()[0],
        'loss': loss,
        'map': 0.0
    }

    print('{} Epoch {} : {:05.3f}%'.format(datatype, ep, metrics_to_return['accuracy']))

    if datatype == 'test':
        for idx, k in enumerate(top_k):
            print(f"Top-{k} Accuracy: {acc[idx].item():.2f}%")
        if opts.dataset == 'le2i':
            matrix.summary()
        matrix.plot()

    return metrics_to_return
