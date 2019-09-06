######## Lenet5 1998 #########
# (correction)setup lenet-5 (7 layers) 原始版本
# alias avg_pooling == ap
# conv1 + ap_1(5*5) + cov2 + ap_2(5*5) + conv_3 + full_1 + output
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, Dropout, ZeroPadding2D

from keras.layers import Layer
from keras import backend as K

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1) # 高斯径向基函数
        res = K.exp(-1 * self.gamma * l2) #
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


def create_lenet(input_shape):
    model = Sequential()
    model.add(ZeroPadding2D(2, input_shape=input_shape))
    print("padding--> ", model.output_shape)
    #1 28×28×6
    # 训练参数计算规则 （cx_size*cy_size*上一层维度（特征图个数）+1）*当前层维度
    # 连接数 当前层和上一层的连接数  训练参数个数*特征图w*特征图h
    # 训练参数 156 = （5×5+1）×6
    # 连接数 122304 = （5×5+1）×6×（28×28）
    model.add(Conv2D(6, (5,5)))#1
    print("conv2d_1--> ", model.output_shape)
    #2 14×14×6
    #训练参数 12 6*(1+1) 当前层数（偏置+采样参数）
    #连接数 5880 = （2×2 + 1）*6*(14*14)
    model.add(AveragePooling2D(pool_size=(2,2)))#2
    print("avg_pooling_1--> ", model.output_shape)
    #3 10×10×16
    #训练参数 1516 = (3*5*5+ 1)*6 + (4*5*5+1)*6 + (4*5*5+1)*3 + (6*5*5+1)
    #连接数 151600 = (3*5*5+ 1)*6×（10×10） + (4*5*5+1)*6×（10×10） + (4*5*5+1)*3*(10*10) + (6*5*5+1)*(10*10)
    model.add(Conv2D(16, (5,5)))#3
    print("conv2d_2--> ", model.output_shape)
    #4 5*5*16
    #训练参数 32 = 16*(1+1)
    #连接个数 2000 = （2*2*1 + 1）*6*(5*5)
    model.add(AveragePooling2D(pool_size=(2,2)))#4
    print("avg_pooling_2--> ", model.output_shape)
    #5 1*1*120
    #训练参数 48120 = (5*5*16 + 1)*120
    #连接个数 48120 = (5*5*16 + 1)*120*(1*1)
    model.add(Conv2D(120, (5,5)))#5
    print("conv2d_3 --> ", model.output_shape)
    model.add(Flatten())
    #6 1*1*84
    #训练参数 10164 = (1*1*120+1)*84
    #连接个数 10164 = (1*1*120+1)*84*(1*1)
    model.add(Dense(84, activation='tanh'))#6
    print("full_1 --> ", model.output_shape)
    #7 1*1*10
    #训练参数 850 = （1*1*84+1)*10(1*1)
    #连接个数 850 = （1*1*84+1)*10(1*1)
    #model.add(Dense(10, activation='sigmoid'))#7
    model.add(RBFLayer(10,0.5))
    print('output --> ', model.output_shape)
    return model


def main():
    model = create_lenet((28, 28, 1))
    # compile
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()


if __name__ == '__main__':
    main()
