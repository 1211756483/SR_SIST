import os
import sys
import time

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from ms_ssimL1 import MS_SSIM_L1_LOSS
import PSNR_utils
from snr_utils import calculate_snr
from ssim_loss import SSIM
from network import SIST
from dataset import *
from L1pro import L1_Charbonnier_loss
from edge_loss import SobelEdgeLoss

def main():
    EPOCH = 500
    BATCH_SIZE = 8
    PATCH_SIZE = 128
    LEARNING_RATE = 0.0002
    lr_list = []
    loss_list = []
    psnr_list = []


    inputPathTrain = r"D:\SIST\data\raw_input"
    inputPathTrain2 = r"D:\SIST\data\rawdata_edge_mix"
    targetPathTrain = r"D:\SIST\data\sr_label"
    inputPathTest = r"D:\SIST\data\testL"
    targetPathTest = r"D:\SIST\data\testH"
    resultPathTest = r"D:\SIST\data\resultTest"

    save_path = r"D:\SIST\loss.txt"

    best_psnr = 0
    best_epoch = 0

    psnr = PSNR_utils.PSNR()  # 实例化峰值信噪比计算类
    psnr = psnr.cuda()


    myNet = SIST(upscale=2, patch_size=1,in_chans=2, img_size=128, window_size=8,
                     img_range=1. ,depths=[6,6,6,6], embed_dim=180, num_heads=[6,6,6,6],
                    mlp_ratio=4, drop_path_rate=0.1,act_cfg=dict(type='GELU'))
    myNet = myNet.cuda()
    # 多卡
    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids) > 1:
        myNet = nn.DataParallel(myNet, device_ids=device_ids)

    criterion1 = SSIM().cuda()  # 结构相似性
    criterion2 = nn.MSELoss().cuda()  # 均方误差
    criterion3 = nn.L1Loss().cuda()
    criterion4 = MS_SSIM_L1_LOSS().cuda()
    criterion5 = SobelEdgeLoss()
    criterion6 = L1_Charbonnier_loss()

    optimizer = optim.Adam(myNet.parameters(), lr=LEARNING_RATE)  # 优化器

    scheduler = MultiStepLR(optimizer, milestones=[300, 400], gamma=0.5) #235


    datasetTrain = MyTrainDataSet(inputPathTrain,inputPathTrain2,targetPathTrain, input_size=128,target_size=256)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE, shuffle=True)

    datasetValue = MyValDataSet(inputPathTest,inputPathTrain2, targetPathTest,  input_size=128,target_size=256)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=1, shuffle=True)


    # 测试数据
    ImageNames = os.listdir(inputPathTest)  # 测试路径文件名
    datasetTest = MyTestDataSet(inputPathTest,inputPathTrain2,targetPathTest)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False)

    print('-------------------------------------------------------------------------------------------------------')
    best_model_path = './model_best.pth'
    if os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    # 定义保存模型的文件夹路径
    save_dir = r"D:\SIST\save_model"
    # 检查保存模型的文件夹是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 打开文件以写入模式
    with open(save_path, "w") as file:
        for epoch in range(EPOCH):
            myNet.train()
            iters = tqdm(trainLoader, file=sys.stdout)
            iters2 = tqdm(valueLoader, file=sys.stdout)
            running_val_loss = 0.0
            epochLoss = 0
            epochSNR = 0
            epochPSNR = 0
            epochSSIM = 0
            timeStart = time.time()
            for index, (x, y) in enumerate(iters, 0):

                input_train, target = Variable(x).cuda(), Variable(y).cuda()

                output_train = myNet(input_train)

                l_ssim = criterion1(output_train, target)
                l_2 = criterion2(output_train, target)  #l2
                l_3 = criterion3(output_train,target)  #l1
                l_4 = criterion4(output_train,target)  #mix
                l_5 = criterion5(output_train,target)
                l_6 = criterion6(output_train,target)  #L1pro
                # 计算PSNR
                output_train = output_train.detach().cpu()  # 将输出张量移动到CPU
                target = target.detach().cpu()  # 将目标张量移动到CPU

                loss = l_6                         #l_4  447   l_5  252
                #loss = 0.6*l_3 + 0.4*0.5*l_5
                #loss = (1 - l_ssim) + l_3
                myNet.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epochLoss += loss.item() # 累加计算本次epoch的loss
                # 计算 SNR
                snr = calculate_snr(target, output_train)
                epochSNR += snr
                # 计算PSNR值
                psnr_value = psnr(output_train, target)
                epochPSNR += psnr_value.item()  # 累加PSNR值
                # 计算当前batch的SSIM
                batch_ssim = criterion1(output_train, target)
                epochSSIM += batch_ssim.item()
                iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f, Batch SNR %.2f, Batch PSNR %.2f, Batch SSIM %.4f'
                                      % (epoch + 1, EPOCH, loss.item(), snr, psnr_value.item(),batch_ssim.item()))
            epochSNR /= len(trainLoader)  # 计算平均 SNR
            # 计算平均PSNR
            epochPSNR /= len(trainLoader)
            epochSSIM /= len(trainLoader)

            print('Epoch %d / %d,  Average Loss %.6f, Average SNR %.2f, Average PSNR %.2f,Average SSIM %.4f'
                  % (epoch + 1, EPOCH, epochLoss / len(trainLoader), epochSNR, epochPSNR, epochSSIM))

            # 将结果写入文件
            file.write("Epoch %d: Average Loss=%.6f, Average PSNR %.2f,Average SSIM=%.4f\n" %
                       (epoch + 1, epochLoss / len(trainLoader), epochPSNR,epochSSIM))

            #网络评估
            myNet.eval()
            psnr_val_rgb = []
            for index, (x, y) in enumerate(valueLoader, 0):
                input_, target_value = x.cuda(), y.cuda()
                with torch.no_grad():
                    output_value = myNet(input_)

                for output_value, target_value in zip(output_value, target_value):
                    psnr_val_rgb.append(psnr(output_value, target_value))
            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            # 保存最佳PSNR的模型权重
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                best_model_path = os.path.join(save_dir, 'model_best.pth')
                torch.save(myNet, best_model_path)

            psnr_list.append(psnr_val_rgb)
            loss_list.append(epochLoss)
            lr_list.append(scheduler.get_last_lr())
            scheduler.step()
            # 保存每个epoch训练完成后的模型权重
            model_path = os.path.join(save_dir, f'model{epoch + 1}.pth')
            torch.save(myNet, model_path)
            timeEnd = time.time()
            print("------------------------------------------------------------")
            print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}.".format(epoch+1, timeEnd-timeStart, epochLoss))
            print('-------------------------------------------------------------------------------------------------------')
        #print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))

        #print('--------------------------------------------------------------')

            # myNet.eval()
            # myNet.cuda()  # 将模型移动到CUDA设备上
            #
            #
            # with torch.no_grad():
            #     timeStart = time.time()
            #     for index, (val_x, val_y) in enumerate(iters2, 0):
            #         torch.cuda.empty_cache()
            #         input_test, target = Variable(val_x).cuda(), Variable(val_y).cuda()
            #         output_test = myNet(input_test)
            #         l_2 = criterion2(output_test, target)  # l2
            #         l_3 = criterion3(output_test, target)  # l1
            #         l_4 = criterion4(output_test, target)  # mix
            #         l_5 = criterion5(output_test, target)
            #         l_6 = criterion6(output_test, target)  # L1pro
            #         running_val_loss += l_4.item()
            # avg_val_loss = running_val_loss / len(valueLoader)
            #
            # iters2.set_description('Testing !!!  Epoch %d / %d,  Batch Loss %.6f'% (epoch + 1, EPOCH, avg_val_loss))
            #
            # print('Epoch %d / %d,  Average Loss %.6f'% (epoch + 1, EPOCH, avg_val_loss))
            #
            #     # 将结果写入文件
            # file.write("Epoch %d: Averagetest Loss=%.6f\n" % (epoch + 1, avg_val_loss))
            #
            # timeEnd = time.time()
            # print('---------------------------------------------------------')
            # print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))

if __name__ == '__main__':
    main()


