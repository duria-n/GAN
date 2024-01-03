# coding:utf8
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import dataloader
import os
import matplotlib.pyplot as plt
import time
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
import matplotlib.pyplot as plt
'''
batch_size=50:
    使用cuda加速下，每代的训练时间大概为11s-16s
    使用cpu进行训练，每代训练时间大概为45-50s
batch_size=120:
'''

import os

# 设置TORCH_HOME环境变量为您希望保存权重的路径
# 设置TORCH_HOME环境变量为您希望保存权重的路径
os.environ['TORCH_HOME'] = r'E:\file\github\GAN\pytorch'


# 接下来，当您使用需要下载预训练权重的函数或库时，权重会被保存到您指定的路径

class Generator(nn.Module):
    def __init__(self, input_shape, input_dim):
        super(Generator, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim

        def block(input_shape, output_shape, alpha=None, momentum=None, dropout=None):
            # 这里是一个列表，所以后面要进行解包操作
            layer = [nn.Linear(input_shape, output_shape)]
            if alpha is not None:
                layer.append(nn.LeakyReLU(alpha))
            if momentum is not None:
                layer.append(nn.BatchNorm1d(output_shape, momentum=momentum))
            if dropout is not None:
                dropout_layer = nn.Dropout(dropout)
                layer.append(dropout_layer)
            return layer

        self.model = nn.Sequential(
            *block(self.input_dim, 256, 0.2, 0.8, 0.3),
            *block(256, 512, 0.2, 0.8, 0.3),
            *block(512, 1024, 0.2, 0.8, 0.3),
            *block(1024, np.prod(self.input_shape)),
            nn.Tanh(),
            nn.Unflatten(1, self.input_shape)
        )
        # print(self.model.state_dict())

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        # print(self.model.state_dict())

    def forward(self, x):
        return self.model(x)


def Data_loader(target_dataset, target_path, target_size, batch_size, shuffle, num_workers, pin_memory, train=True):
    transform = transforms.Compose(
        [transforms.Resize(target_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    os.makedirs('./data/test_image', exist_ok=True)
    train_dataset = target_dataset(target_path, train=train, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers, pin_memory=pin_memory)
    return data_loader


def Test_image(model, row, clo, input_dim, image_num, epoch=None):
    model.eval()
    if epoch is not None:
        os.makedirs('./data/test_image', exist_ok=True)
    else:
        os.makedirs('./data/predict_image', exist_ok=True)
    with torch.no_grad():
        noise = torch.randn(image_num, input_dim)
        noise = noise.to(device)
        gen_imgs = model(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = gen_imgs.cpu()
        fig, axs = plt.subplots(row, clo)
        cnt = 0
        for i in range(row):
            for j in range(clo):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                cnt += 1
        if epoch is not None:
            fig.savefig('./data/test_image/mnist_%d.png' % epoch)
        else:
            fig.savefig('./data/predict_image/minst.png')
        plt.close()


def Train(epochs, batch_size, input_dims, lr):
    # 参数初始化
    epoch_digits = len(str(epochs))
    target_dataset = datasets.MNIST
    target_path = './data'
    batch_size = batch_size
    num_workers = 16
    input_dim = 100
    input_image_shape = (28, 28, 1)
    output_shape = 1
    # 模型初始化
    # 生成模型初始化并移到GPU
    gen_model = Generator(input_image_shape, input_dim)
    gen_optim = torch.optim.Adam(gen_model.parameters(), lr=lr, betas=(0.5, 0.002), weight_decay=1e-5)
    gen_model.to(device)
    # 分辨模型初始化并移到GPU
    discriminator_model = Discriminator(input_shape=input_image_shape, output_shape=output_shape)
    dis_optim = torch.optim.Adam(discriminator_model.parameters(), lr=lr, betas=(0.5, 0.002), weight_decay=1e-5)
    discriminator_model.to(device)
    loss = torch.nn.BCELoss()

    # 数据初始化，pin_memory加快cpu到gpu的数据传输速度，num_workers增加数据读取
    data_loader = Data_loader(target_dataset=target_dataset, target_path=target_path, target_size=28,
                              batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    log = []
    best_para = gen_model.state_dict()
    best_fid = torch.finfo(torch.float64).max
    flag = 0
    # 开始训练
    for epoch in range(epochs):
        start_time = time.time()
        real_image_path = os.path.join(r'E:\file\github\GAN\pytorch', 'fid_image', 'real_image', str(epoch + 1))
        fake_image_path = os.path.join(r'E:\file\github\GAN\pytorch', 'fid_image', 'fake_image', str(epoch + 1))
        for i, (image, _) in enumerate(data_loader):
            real_image = image
            real_label = torch.ones(image.size(0), 1)
            fake_label = torch.zeros(image.size(0), 1)
            real_image = real_image.to(device)
            real_label = real_label.to(device)
            fake_label = fake_label.to(device)
            # 训练生成器
            gen_optim.zero_grad()
            noise = torch.randn(image.size(0), input_dims)
            noise = noise.to(device)
            gen_images = gen_model(noise)
            g_loss = loss(discriminator_model(gen_images), real_label)
            g_loss.backward()
            gen_optim.step()

            # 训练鉴别器，这款里训练鉴别器时需要重新生成fake_image，否则loss反向传播报错
            dis_optim.zero_grad()
            dis_loss = loss(discriminator_model(real_image), real_label)
            # gen_optim.zero_grad()
            noise = torch.randn(image.size(0), input_dims)
            noise = noise.to(device)
            gen_images = gen_model(noise)
            fake_loss = loss(discriminator_model(gen_images), fake_label)
            total_dis_loss = 0.5 * dis_loss + 0.5 * fake_loss
            total_dis_loss.backward()
            # 这里只训练鉴别器，因此参数更新只需要更新鉴别器参数
            dis_optim.step()
            if i > (9 * len(data_loader) / 10):
                os.makedirs(real_image_path, exist_ok=True)
                to_pil = transforms.ToPILImage()
                for j, image in enumerate(real_image):
                    pil_real_image = to_pil(image.cpu())
                    # image.save('fid_image/real_image/image%d.png'%i+1
                    pil_path = os.path.join(real_image_path,f'{i}_real_image_{j + 1}.png')
                    pil_real_image.save(pil_path)
                os.makedirs(f'fid_image/fake_image/{epoch + 1}', exist_ok=True)
                gen_images = gen_images.permute(0, 3, 1, 2)
                for j, image in enumerate(gen_images):
                    pil_path = os.path.join(fake_image_path,f'{i}_fake_image_{j + 1}.png')
                    pil_fake_image = to_pil(image.cpu())
                    pil_fake_image.save(pil_path)
        curr_fid = calculate_fid_given_paths([real_image_path, fake_image_path],
                                                    batch_size=batch_size,device=device,dims=2048)
        if curr_fid < best_fid:
            best_fid = curr_fid
            best_para = gen_model.state_dict()
            flag = 0
        else:
            flag += 1
        dur_time = time.time() - start_time
        log.append([curr_fid, dur_time])
        print(f"第{epoch + 1:{epoch_digits}d}轮  curr_fid:{log[-1][0]:.6f}, dur_time:{dur_time:.3f}")
        if flag > 1000 or epoch == epochs - 1:
            torch.save(best_para,'gen_model.pt')
            break
        # print(f"第{epoch + 1:{epoch_digits}d}轮 g_loss: {g_loss.item():.6f}, "
        #       f"dis_loss: {dis_loss.item():.6f}, "
        #       f"fake_loss: {fake_loss.item():.6f}, "
        #       f"total_loss: {total_dis_loss.item():.6f},dur_time:{dur_time:.5f}")
        # curr_fid = calculate_fid_given_paths([real_image,gen_images],batch_size=16,cuda=True)
        # # print("第%d轮g_loss:%f, dis_loss:%f, fake_loss%f, total_loss:%f"%(epoch+1, log[i][0], log[i][1], log[i][2], log[i][3]))
        # if curr_fid < best_fid:
        #     flag = 0
        #     best_loss = curr_fid
        #     best_para = gen_model.state_dict()
        # else:
        #     flag += 1
        # #这里早停不太对，模型保存时间也不太对
        # if flag > 2000:
        #     torch.save(best_para, 'gen_model.pt')
        #     break
        if epoch == epochs // 2:
            lr = 1e-6
            gen_optim.param_groups[0]['lr'] = lr
            dis_optim.param_groups[0]['lr'] = lr
        if epoch % 200 == 0:
            model = Generator(input_image_shape, input_dim)
            model.load_state_dict(best_para)
            model.to(device)
            Test_image(model=model, row=5, clo=5, input_dim=input_dim, image_num=25, epoch=epoch)
    epochs = range(1, len(log) + 1)
    fids = [entry[0] for entry in log]  # 获取所有FID得分
    plt.scatter(epochs, fids, label='FID')
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.title('FID Score Over Epochs')
    plt.legend()
    plt.savefig("fid_score.png")
    print(gen_model.state_dict())


def Predict(model_path, input_dim, input_shape):
    model = Generator(input_shape, input_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    Test_image(model, 5, 5, input_dim, 25)


if __name__ == "__main__":
    epochs = 10000
    batch_size = 220
    lr = 1e-5
    input_dim = 100
    # 如果cuda可用，则使用cuda加速训练
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    Train(epochs, batch_size, input_dim, lr)
    Predict('./gen_model.pt', input_dim, (28, 28, 1))
