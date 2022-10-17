import sys
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Reshape, LeakyReLU,MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import numpy as np

image_size = 64
im = cv2.imread('./data/faces/2.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im,(64,64))
im = np.array(im)
trainimg = np.expand_dims(im,0)

imglist = os.listdir('./data/faces')
for i in range(len(imglist)):
    im = cv2.imread('./data/faces/%s'%imglist[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(64,64))
    im = np.array(im)
    trainimg = np.concatenate((trainimg,np.expand_dims(im,0)),0)
    if i % 1000 ==0:
        print(i)
    if i == 5000:
        break
print(trainimg.shape)


def generator_model():
    model = Sequential()
    model.add(Flatten(input_shape = (8, 8, 3)))
    model.add(Dense(8*8*64, use_bias=False))
    model.add(BatchNormalization())
    model.add(Reshape((8, 8, 64)))  # output: 8*8*256
    model.add(Conv2DTranspose(256, 5, strides=2, padding='SAME'))  # output: (None,16,16,128)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(128, 3, strides=2, padding='SAME'))  # output: (None, 32, 32, 64)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(3, 3, strides=2, padding='SAME'))  # output: (None, 64, 64, 3)
    model.add(Activation('sigmoid'))
    return model

def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, padding='SAME', kernel_size=5, strides=2, input_shape=(image_size, image_size, 3)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Conv2D(128, padding='SAME', kernel_size=5, strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2D(256, padding='SAME', kernel_size=5, strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv2D(128, padding='SAME', kernel_size=5, strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
# g = discriminator_model()
# g.summary()
# sys.exit(0)



def combine(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)

    return model


def combine_images(images):
    num = images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1], 3),
                     dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0:3] = img[:, :, :]

    return image


result_path = './result/'

if os.path.exists(result_path) == False:
    os.makedirs(result_path)

model_path = './h5/'
generated_image_path = './generated_image/'

if os.path.exists(model_path)==False:
    os.makedirs(model_path)
if os.path.exists(generated_image_path)==False:
    os.makedirs(generated_image_path)

def generated(noise_need, name):
    g = generator_model()
    try:
        g.load_weights(model_path + "generatorA")
        print("生成器权重导入成功")
    except:
        print("无权重")
    noise_need = np.random.normal(0, 1, size=(1, 8,8,3))
    generated_image_need = g.predict(noise_need, verbose=0)
    image = combine_images(generated_image_need)
    image = image * 255
    Image.fromarray(image.astype(np.uint8)).save(
        result_path + name + ".png")


def train(BATCH_SIZE, X_train):
    # 生成图片的连接图片数
    generated_image_size = 36
    # 读取图片
    X_train = ((X_train.astype(np.float32)) - 0) / 255

    # 模型及其优化器
    d = discriminator_model()
    g = generator_model()
    g_d = combine(g, d)
    d_optimizer = Adam()
    g_optimizer = Adam()
    g.compile(loss='binary_crossentropy', optimizer='adam')  # 生成器
    g_d.compile(loss='binary_crossentropy', optimizer=g_optimizer)  # 联合模型
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optimizer)  # 判别器

    # 导入权重
    try:
        d.load_weights(model_path + "discriminatorA")
        print("判别器权重导入成功")
        g.load_weights(model_path + "generatorA")
        print("生成器权重导入成功")
    except:
        print("无权重")

    for epoch in range(1000):
        # 每1轮打印一次当前轮数
        if epoch % 1 == 0:
            print('Epoch is ', epoch)
        for index in range(X_train.shape[0] // BATCH_SIZE):
            # 产生（0，1）的正态分布的维度为（BATCH_SIZE, character）的矩阵
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, 8*8*3))
            noise = noise.reshape((BATCH_SIZE,8,8,3))
            train_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_image = g.predict(noise, verbose=0)

            if index % 50 == 0:
                # 每50次输出一次图片
                noise_need = np.random.normal(0, 1, size=(generated_image_size, 8,8,3))
                generated_image_need = g.predict(noise_need, verbose=0)
                image = combine_images(generated_image_need)
                image = image * 255
                Image.fromarray(image.astype(np.uint8)).save(
                    generated_image_path + str(epoch) + "_" + str(index) + ".png")
            # 每运行一次训练一次判别器
            if index % (2) == 0:
                X = np.concatenate((train_batch, generated_image))
                Y = list((np.random.rand(BATCH_SIZE) * 10 + 90) / 100) + [0] * BATCH_SIZE
                Y = np.array(Y)
                Y = Y.reshape((X.shape[0],1))
                d_loss = d.train_on_batch(X, Y)


            noise = np.random.normal(0, 1, size=(BATCH_SIZE, 8*8*3))
            noise = noise.reshape((BATCH_SIZE,8,8,3))
            d.trainable = False
            Y = list((np.random.rand(BATCH_SIZE) * 10 + 90) / 100)
            Y = np.array(Y)
            Y = Y.reshape((BATCH_SIZE, 1))
            g_loss = g_d.train_on_batch(noise, Y)
            d.trainable = True
            if index % 10 == 0:
                print('batch: %d, g_loss: %f, d_loss: %f' % (index, g_loss, d_loss))

        g.save(model_path+'g.h5', save_format='h5')
        print('Successfully save generatorA')
        # d.save(model_path + 'd.h5', save_format='h5')
        # print('Successfully save discriminatorA')


train(BATCH_SIZE=64, X_train=trainimg)
