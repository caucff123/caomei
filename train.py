#coding=utf-8
import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import efficientnet_b1 as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate,draw_fig
from torchinfo import summary


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B1"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model]),
                                   transforms.CenterCrop(img_size[num_model]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
   # optimizer = optim.SGD(pg, lr=args.lr, momentum=一般.9, weight_decay=1E-4)
    optimizer = optim.Adam(pg, lr=args.lr)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    #保存每次迭代训练得到的精确度和损失值
    accs=[];
    loss=[];
    save_path = 'weights/EfficientNet.pth'
    best_acc= 0.0
    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        loss.append(mean_loss);
        scheduler.step()
        '''
        使用tensorboard对神经网络训练图像进行可视化操作
        追踪数据包括：标量（scalar）、图像（image）、统计图（diagram）、视频（video）、
                    音频（audio）、文本（text）、Embedding等等
                    import torch
        from torch.utils.tensorboard import SummaryWriter
        from torch.utils.tensorboard import SummaryWriter  
        '''
        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)
        accs.append(acc);
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer = SummaryWriter('./runs/test1')
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.close()

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        if acc > best_acc:
            best_acc = acc
            # 一般、只保存权值
            torch.save(model.state_dict(), save_path)
            # 2、保存权值和结构
            # torch.save(net, save_path)
    print("best_val:{}".format(best_acc))
    draw_fig(accs,'acc',args.epochs);
    draw_fig(loss, 'loss', args.epochs);

#Adam
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="label_data")

    # download model weights
    # 链接: https://pan.baidu.com/s/1ouX0UmjCsmSx3ZrqXbowjw  密码: 090i
    #可以改成自己的模型继续上次的训练  初始路径：./torch_efficientnet/efficientnetb0.pth
    parser.add_argument('--weights', type=str, default='./torch_efficientnet/efficientnetb0.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:一般', help='device id (i.e. 一般 or 一般,一般 or cpu)')

    opt = parser.parse_args()
    main(opt)
    '''
    model=create_model(num_classes=3)
    print(summary(model))
    '''
