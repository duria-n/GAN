from __future__ import print_function, division

import os

from keras.datasets import mnist
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np


class Gan():
    def __init__(self):
        self.img_row = 28
        self.img_clos = 28
        self.img_channels = 1
        self.img_shape = (self.img_row,self.img_clos,self.img_channels)
        self.latent_dim = 100
        #平滑过程中梯度的更新参数，越小变化越快
        optimizer = Adam(0.0002,0.5)

        #build and compile the dicsriminator
        self.discriminator = self.build_discriminator()
        #model.compile配置模型
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,metrics=['accuracy'])

        #build and compile generator
        self.generator = self.build_generator()

        # generator take noise as input and generator image
        #Input(shape,batch_size,dtype),这里表示输入是一个一维向量，其长度为self.latent_dim
        Z = Input(shape=(self.latent_dim,))
        img = self.generator(Z)

        #only train generator as goal

        self.discriminator.trainable = False

        validity = self.discriminator(img)
        #潜在空间的输入 Z 映射到 validity这个组合模型用于训练生成器
        self.combined = Model(Z,validity)
        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer)



    def build_generator(self):
        model = Sequential(name='Generator')

        model.add(Dense(256,input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        #较大的动量参数会使得更新方向更平滑，较小的动量参数会使得收敛更快，但可能会导致出现局部最小值
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        #np.prod()返回参数相乘的结果
        model.add(Dense(np.prod(self.img_shape),activation='tanh'))
        #将输出的一维张量reshape成和self.shape相同大小的矩阵
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        # 创建一个 Model 对象，该对象将输入噪声映射到生成的图像，并返回这个模型
        return Model(noise,img)

    def build_discriminator(self):
        model =Sequential(name='Discriminator')

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        #这里就是一个简单的而分类问题
        model.add(Dense(1,activation='sigmoid'))
        model.summary()


        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img,validity)


    def train(self,epochs,batch_size=128,sample_interval=50):

        #load  dataset
        (X_train,_),(_,_) = mnist.load_data()

        X_train = X_train/127.5 - 1
        #np.expand_dims(X_train,axis=3)只能在第三个维度上额外增加一个值为1的维度
        # np.broadcast_to(X_train, (*X_train.shape[:-1], 3))  将新的维度大小改为 3
        X_train = np.expand_dims(X_train,axis=3)

        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        for epoch in range(epochs):

            idx = np.random.randint(0,X_train.shape[0],batch_size)
            imgs = X_train[idx]

            noise = np.normal(0,1,(batch_size,self.latent_dim))

            gen_imgs = self.generator.predict(noise)

            # Train discriminator，input：gen_imgs
            d_loss_real = self.discriminator.train_on_batch(imgs,valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,fake)

            d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)

            # Train generator，new noise
            noise = np.random.normal(0,1,(batch_size,self.latent_dim))
            gen_loss = self.combined.train_on_batch(noise,valid)

            #plot
            print("%d [D loss:%.f, acc:%.2f%%] [G loss:%f]" % (epoch,d_loss[0],
                                                               100*d_loss[1],gen_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)


    def sample_images(self,epoch):
        # plt.ion()
        r,c = 5,5
        #保存图片的时候重新随机生成了一波数据，和之前训练生成器时生成的数据没有太大的关系
        noise = np.random.normal(0,1,(r*c,self.latent_dim))
        gen_imgs = self.generator.predict(noise)


        gen_imgs = 0.5 * gen_imgs + 0.5
        fig,axs = plt.subplots(r,c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')
                cnt+=1
        fig.savefig("./images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan  = Gan()
    gan.train(epochs=2000,batch_size=32,sample_interval=400)
