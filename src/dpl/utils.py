import cv2
import matplotlib.pyplot as plt
import numpy as np

# 棋盘显示图片
# images 输入图片 eg:（600， 256， 256， 3）
# grids 棋盘格宽高
def showImages(images, grids=(5,5)):
    cell_nums = grids[0]*grids[1]
    if images.shape[0] < cell_nums:
        return
    print("info:\nimage size --> ", images[0].shape[0], "x", images.shape[1])
    if len(images.shape) == 4:
        print("channels --> ", images.shape[3])
    plt.figure(figsize=(10,10))
    for i in range(cell_nums):
        plt.subplot(grids[0],grids[1],i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])#, cmap=plt.cm.binary)
    plt.show()

# alexnet 网络预处理
# resize image to (227, 227, 3)
def preprocess4Alexnet(input_data, sample_nums, dsize=(227, 227)):
    if sample_nums >= input_data.shape[0]: 
        return;
    converted = np.zeros((sample_nums, dsize[0], dsize[1], 3), dtype=np.float32)
    for i in range(sample_nums):
        res = input_data[i]
        res = res.astype('float32')
        # 修改图片尺寸
        res = cv2.resize(res, dsize)
        if len(res.shape) == 2: #如果传入的是灰度图，则转换成rgb
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR);
        converted[i] = res;
    return converted;

# 显示loss曲线
def showAccLossCurve(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
