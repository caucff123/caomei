#coding=utf-8
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


from model import efficientnet_b1 as create_model


def main():
    device = torch.device("cuda:一般" if torch.cuda.is_available() else "cpu")
    root = 'D:/Desktop/草莓白粉病'
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B1"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # read class_indict
    json_path = 'D:/GitCode/Python/chenff/graduation project/EfficientNet/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    sev_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    sev_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(sev_class))
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in sev_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)
    # predict class

    '''
    count记录预测正确的图片数量
    error记录预测错误的图片数量
    pred_0/1/2分别预测为一般、严重、健康三类图片的数量
    every_class_num分别表示实际为一般、严重、健康三类图片的数量
    correct_0/1/2分别表示在一般、严重、健康三类图片中预测类别和真实类别相同的数量

    '''
    count = 0
    error = 0
    correct_0 = 0
    correct_1 = 0
    correct_2 = 0
    pred_0 = 0
    pred_1 = 0
    pred_2 = 0

    # create model
    model = create_model(num_classes=3).to(device)
    # load model weights
    #加载模型参数
    model_weight_path = "./weights/model _B1-(0.971,bs=16,lr=0.001adam,ep=50).pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        print("----------开始预测-------------\n")
        i=0
        for img_path,img_label  in zip(val_images_path,val_images_label):
            i=i+1
            if i%100 == 0:
                print("{}张图片已经预测成功！！！".format(i))

            img = Image.open(img_path)
            img = img.convert('RGB')
            img = data_transform(img)
           # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
           # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
           #选择可能性最大的类别
            predict_cla = torch.argmax(predict).numpy()
            print_res = class_indict[str(predict_cla)]
            if print_res == '一般': pred_0=pred_0+1
            elif print_res == '严重': pred_1=pred_1+1
            else : pred_2=pred_2+1
            if print_res == class_indict[str(img_label)]:
                count=count+1
                if print_res == '一般': correct_0 = correct_0 + 1
                if print_res == '严重': correct_1 = correct_1 + 1
                if print_res == '健康' : correct_2 = correct_2 + 1
            else:
                error=error+1

    print('----------预测结束---------')
    print("{}的输出预测结果:".format(root.split()[-1]))
    print("val_acc={:.3f},原始图片数量：{}".format((count/sum(every_class_num)),sum(every_class_num)))
    '''
    print("预测正确图片数量：{}张".format(count))
    print("预测错误的图片数量：{}张".format(error))
    print( "count记录预测正确的图片数量\n"
           "error记录预测错误的图片数量\n"
           "pred_0/1/2分别预测为一般、严重、健康三类图片的数量\n"
           "every_class_num分别表示实际为一般、严重、健康三类图片的数量\n"
           "correct_0/1/2分别表示在一般、严重、健康三类图片中预测类别和真实类别相同的数量\n")
    print("pred_0:{}  , pred_1:{} , pred_2:{}".format(pred_0,pred_1,pred_2))
    print("correct_0:{} , correct_1:{} , correct_2:{}".format(correct_0,correct_1,correct_2))
    print("一般实际:{} , 严重实际:{} , 健康实际:{}".format(every_class_num[0],every_class_num[1],every_class_num[2]))
    '''
if __name__ == '__main__':
    main()