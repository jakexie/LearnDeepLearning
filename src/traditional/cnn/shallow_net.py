# setup shallow network 一个隐藏层
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD


def create_mlp_net(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape)) # 展开
    print(model.output_shape)
    model.add(Dense(1024, activation='relu')) # 全连接(隐藏层1)
    print(model.output_shape)
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax')) # 输出
    print(model.output_shape)
    return model


def main():
    model = create_mlp_net((28,28))
    sgd=SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=["accuracy"]) # 编译
    model.summary()


if __name__ == '__main__':
    main()
