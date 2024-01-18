# coding:utf8
import random
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import dataloader
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_fid.fid_score import calculate_fid_given_paths
import matplotlib.pyplot as plt
import signal

"""
cuda加速下随着负载的增加，加速效果越来越明显
"""

# 设置望保存权重的路径
os.environ["TORCH_HOME"] = r"E:\file\github\GAN\pytorch"


def set_seed(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(torch.initial_seed())
        np.random.seed()
        random.seed()
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch.initial_seed())
            torch.cuda.manual_seed(torch.initial_seed())
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


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
            nn.Unflatten(1, self.input_shape),
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
            nn.Sigmoid(),
        )
        # print(self.model.state_dict())

    def forward(self, x):
        return self.model(x)


def Data_loader(
    target_dataset,
    target_path,
    target_size,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    train=True,
):
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    os.makedirs("./data/test_image", exist_ok=True)
    train_dataset = target_dataset(
        target_path, train=train, download=True, transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return data_loader


def Test_image(model, row, clo, input_dim, image_num, epoch=None):
    model.eval()
    if epoch is not None:
        os.makedirs("./data/test_image", exist_ok=True)
    else:
        os.makedirs("./data/predict_image", exist_ok=True)
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
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                cnt += 1
        plt.show()
        if epoch is not None:
            fig.savefig("./data/test_image/mnist_%d.png" % epoch)
        else:
            fig.savefig("./data/predict_image/minst.png")
        plt.close()


class Trainer:
    def __init__(
        self,
        gen_model,
        dis_model,
        start_epoch,
        end_epoch,
        best_fid,
        batch_size,
        lr,
        input_dims,
        log,
    ):
        self.gen_model = gen_model
        self.dis_model = dis_model
        self.dis_optim = torch.optim.Adam(
            self.dis_model.parameters(), lr=lr, betas=(0.5, 0.002), weight_decay=1e-5
        )
        self.dis_lr_scheduler = ReduceLROnPlateau(self.dis_optim, "min", 0.1, 50)
        self.gen_optim = torch.optim.Adam(
            gen_model.parameters(), lr=lr, betas=(0.5, 0.002), weight_decay=1e-5
        )
        self.gen_lr_scheduler = ReduceLROnPlateau(self.gen_optim, "min", 0.1, 50)
        self.curr_epoch = start_epoch
        self.total_epoch = end_epoch
        self.best_fid = best_fid
        self.best_gen_para = gen_model.state_dict()
        self.dis_para = dis_model.state_dict()
        self.batch_size = batch_size
        self.curr_lr = lr
        self.input_dims = input_dims
        self.log = log
        self.flag = 0
        self.input_shape = (28, 28, 1)
        self.ckpt_dir = r"E:\file\github\GAN\pytorch\check_point"
        signal.signal(signal.SIGINT, self.signal_handler)

    #   'epoch': self.curr_epoch + 1,
    #                 'state_dict': self.gen_model.state_dict(),
    #                 'best_fid': self.best_fid,
    #                 'optimizer': self.gen_optim.state_dict(),
    #                 'state_dict2': self.dis_model.state_dict(),
    #                 'optimizer2': self.dis_optim.state_dict()
    def load_weight(self, checkpoint_path):
        checkpoints = torch.load(checkpoint_path, map_location=device)
        self.gen_model.load_state_dict(checkpoints["state_dict"])
        self.dis_model.load_state_dict(checkpoints["state_dict1"])
        self.gen_optim.load_state_dict(checkpoints["optimizer"])
        self.dis_optim.load_state_dict(checkpoints["optimizer1"])
        self.dis_lr_scheduler.load_state_dict(checkpoints['lr1'])
        self.gen_lr_scheduler.load_state_dict(checkpoints['lr'])
        self.best_fid = checkpoints["best_fid"]
        self.start_epoch = checkpoints["epoch"]
        self.flag = checkpoints["flag"]
        # self.flag = 500

    def get_timestamp(self):
        return time.strftime("%Y%m%d-%H%M%S")

    def save_checkpoints(self, filename):
        try:
            checkpoints = {
                "epoch": self.curr_epoch + 1,
                "state_dict": self.gen_model.state_dict(),
                "best_fid": self.best_fid,
                "optimizer": self.gen_optim.state_dict(),
                "state_dict1": self.dis_model.state_dict(),
                "optimizer1": self.dis_optim.state_dict(),
                "flag": self.flag,
                "lr": self.dis_lr_scheduler.state_dict(),
                "lr1": self.gen_lr_scheduler.state_dict(),
            }
            filename = os.path.join(self.ckpt_dir, filename)
            torch.save(checkpoints, filename)
            print(f"checkpoint saved at {filename}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def save_best_checkpoints(self, filename):
        try:
            checkpoints = {
                "epoch": self.curr_epoch + 1,
                "state_dict": self.gen_model.state_dict(),
                "best_fid": self.best_fid,
                "optimizer": self.gen_optim.state_dict(),
                "state_dict1": self.dis_model.state_dict(),
                "optimizer1": self.dis_optim.state_dict(),
                "flag": self.flag,
                 "lr": self.dis_lr_scheduler.state_dict(),
                "lr1": self.gen_lr_scheduler.state_dict(),
            }
            filename = os.path.join(self.ckpt_dir, filename)
            torch.save(checkpoints, filename)
            print(f"checkpoint saved  to {filename}")
        except Exception as e:
            print(f"Error saving best checkpoint: {e}")

    def signal_handler(self, signal, frame):
        self.save_checkpoints(f"curr_checkpoint_Inter.pth.tar")
        self.save_best_checkpoints(f"best_checkpoint_Inter.pth.tar")

    def plt_fig(self):
        epochs = range(1, len(self.log) + 1)
        fids = [entry[0] for entry in self.log]  # 获取所有FID得分
        plt.scatter(epochs, fids, label="FID")
        plt.xlabel("Epoch")
        plt.ylabel("FID Score")
        plt.title("FID Score Over Epochs")
        plt.legend()
        plt.savefig("fid_score.png")

    def train(self):
        try:
            epoch_digits = len(str(self.total_epoch))
            epochs = self.total_epoch
            target_dataset = datasets.MNIST
            target_path = "./data"
            num_workers = 24
            # lr = self.curr_lr
            input_dim = self.input_dims
            input_image_shape = (28, 28, 1)
            # output_shape = 1
            gen_model = self.gen_model
            gen_optim = self.gen_optim
            gen_model.to(device)
            # 分辨模型初始化并移到GPU
            discriminator_model = self.dis_model
            dis_optim = self.dis_optim
            discriminator_model.to(device)
            loss = torch.nn.BCELoss()
            # 数据初始化，pin_memory加快cpu到gpu的数据传输速度，num_workers增加数据读取
            data_loader = Data_loader(
                target_dataset=target_dataset,
                target_path=target_path,
                target_size=28,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

            # 开始训练
            for epoch in range(self.start_epoch, epochs):
                start_time = time.time()
                real_image_path = os.path.join(
                    r"E:\file\github\GAN\pytorch",
                    "fid_image",
                    "real_image",
                    str(epoch + 1),
                )
                fake_image_path = os.path.join(
                    r"E:\file\github\GAN\pytorch",
                    "fid_image",
                    "fake_image",
                    str(epoch + 1),
                )
                for i, (image, _) in enumerate(data_loader):
                    real_image = image
                    real_label = torch.ones(image.size(0), 1)
                    fake_label = torch.zeros(image.size(0), 1)
                    real_image = real_image.to(device)
                    real_label = real_label.to(device)
                    fake_label = fake_label.to(device)
                    # 训练生成器
                    gen_optim.zero_grad()
                    noise = torch.randn(image.size(0), input_dim)
                    noise = noise.to(device)
                    gen_images = gen_model(noise)
                    g_loss = loss(discriminator_model(gen_images), real_label)
                    g_loss.backward()
                    gen_optim.step()

                    # 训练鉴别器，这款里训练鉴别器时需要重新生成fake_image，否则loss反向传播报错
                    dis_optim.zero_grad()
                    dis_loss = loss(discriminator_model(real_image), real_label)
                    # gen_optim.zero_grad()
                    noise = torch.randn(image.size(0), input_dim)
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
                            pil_path = os.path.join(
                                real_image_path, f"{i}_real_image_{j + 1}.png"
                            )
                            pil_real_image.save(pil_path)
                        os.makedirs(f"fid_image/fake_image/{epoch + 1}", exist_ok=True)
                        gen_images = gen_images.permute(0, 3, 1, 2)
                        for j, image in enumerate(gen_images):
                            pil_path = os.path.join(
                                fake_image_path, f"{i}_fake_image_{j + 1}.png"
                            )
                            pil_fake_image = to_pil(image.cpu())
                            pil_fake_image.save(pil_path)
                curr_fid = calculate_fid_given_paths(
                    [real_image_path, fake_image_path],
                    batch_size=batch_size,
                    device=device,
                    dims=2048,
                )
                self.curr_epoch = epoch
                if curr_fid < self.best_fid:
                    self.best_fid = curr_fid
                    self.best_gen_para = gen_model.state_dict()
                    self.flag = 0
                    self.save_best_checkpoints(f"best_checkpoint.pth.tar")
                else:
                    self.flag += 1
                dur_time = time.time() - start_time
                self.log.append([curr_fid, dur_time])
                print(
                    f"第{epoch + 1:{epoch_digits}d}轮  curr_fid:{self.log[-1][0]:.6f}, best_fid:{self.best_fid:.6f} dur_time:{dur_time:.3f}"
                )
                if self.flag > 1000 or epoch == epochs - 1:
                    self.save_checkpoints(f"final_checkpoint.pth.tar")
                    break

                if epoch == epochs // 2:
                    lr = 1e-6
                    gen_optim.param_groups[0]["lr"] = lr
                    dis_optim.param_groups[0]["lr"] = lr
                if epoch % 200 == 0:
                    model = Generator(input_image_shape, input_dim)
                    model.load_state_dict(self.best_gen_para)
                    model.to(device)
                    Test_image(
                        model=model,
                        row=5,
                        clo=5,
                        input_dim=input_dim,
                        image_num=25,
                        epoch=epoch,
                    )
                self.dis_lr_scheduler.step(g_loss)
                self.gen_lr_scheduler.step(total_dis_loss)
            print(gen_model.state_dict())
        except KeyboardInterrupt:
            print("Training interrupted. Saving checkpoints...")
            self.save_checkpoints(f"curr_checkpoint_Inter.pth.tar")
            self.save_best_checkpoints(f"best_checkpoint_Inter.pth.tar")

        except Exception as e:
            # 捕获其他异常
            print(f"An error occurred: {e}")
            self.save_checkpoints(f"curr_checkpoint_Err.pth.tar")
            self.save_best_checkpoints(f"best_checkpoint_Err.pth.tar")

        # 参数初始化

class Predict_data():
    def __init__(self, model_path, input_dims, input_image_shape):
        self.model_path = model_path
        self.input_dims = input_dims
        self.input_image_shape = input_image_shape
        self.model = Generator(self.input_image_shape, self.input_dims).to(device)
        self._load_weight()
        Test_image(self.model, 5, 5, self.input_dims, 25)
        
        
    def _load_weight(self):
        checkpoints = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(checkpoints["state_dict"])
        # self.gen_optim.load_state_dict(checkpoints["optimizer"])
      
        
# def Predict(model_path, input_dims, input_image_shape):
#     model = Generator(input_image_shape, input_dims).to(device)
#     model.load_state_dict(torch.load(model_path))
#     Test_image(model, 5, 5, input_dims, 25)


class GAN(nn.Module):
    def __init__(self, input_image_shape, input_dims, output_shape, device):
        super(GAN, self).__init__()
        self.gen_model = Generator(input_image_shape, input_dims).to(device)
        self.dis_model = Discriminator(input_image_shape, output_shape).to(device)


if __name__ == "__main__":
    # 模型初始化
    set_seed(10101)
    epochs = 10000
    batch_size = 192
    lr = 1e-5
    input_dims = 100
    input_shape = (28, 28, 1)
    output_shape = 1
    log = []
    best_fid = torch.finfo(torch.float64).max
    # 如果cuda可用，则使用cuda加速训练
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    gan_model = GAN(input_shape, input_dims, output_shape, device)
    gan_train = Trainer(
        gan_model.gen_model,
        gan_model.dis_model,
        0,
        10000,
        best_fid,
        batch_size,
        lr,
        input_dims,
        log,
    )
    if os.listdir(".\check_point"):
        checkpoint_path = (
            r".\check_point/best_checkpoint_Inter.pth.tar"
        )
        gan_train.load_weight(checkpoint_path)
    gan_train.train()
    set_seed()
    predict_image =  Predict_data(r".\check_point\final_checkpoint.pth.tar", input_dims, input_shape)
    # set_seed()
