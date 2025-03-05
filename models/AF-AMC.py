import torch
import tensorflow as tf
import os,json, datetime
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"] = "device=cuda,floatX=float32"
import numpy as np
import tensorflow.keras.models as models
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,LSTM
import matplotlib.pyplot as plt
import pickle as cPickle
from matplotlib import font_manager
times_new_roman_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_manager.fontManager.addfont(times_new_roman_path)
plt.rcParams['font.family'] = 'Times New Roman'
root_dir = "/home/baolin/PycharmProjects/AF-AMC"




print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Load the dataset ...
#  You will need to seperately download or generate this file
dataFile = f"{root_dir}/data/RML2016.10a_dict.pkl"
with open(dataFile, 'rb') as f:
    Xd = cPickle.load(f, encoding='latin1')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
# print('torch GPU:', torch.cuda.is_available())
# print('tensorflow GPU:', tf.test.is_gpu_available())
#   以字典形式存储数据
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
#   设置随机种子（确保每次运行代码选择的数据划分都是一样的）
np.random.seed(2016)
#   定义数据集大小
n_examples = X.shape[0]
#   取一半的数据集作为训练集
n_train = n_examples * 0.6
#   选择训练和测试索引
train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]


#   将标签转换为one-hot编码形式
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1


Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
#   从训练数据中提取应该输入的形状并打印出来
in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods

# Build VT-CNN2 Neural Net model using Keras primitives --
#  - Reshape [N,2,128] to [N,1,2,128] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization
#   构建网络模型，由两个卷积层，两个全连接层，组成，并引入丢弃层防止过拟合
dr = 0.5 # dropout rate (%)
model = models.Sequential()
model.add(Reshape(in_shp+[1], input_shape=in_shp))
model.add(ZeroPadding2D((0,2),data_format="channels_last"))
model.add(Convolution2D(128, (1, 3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform',data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None))
model.add(ZeroPadding2D((0, 2),data_format="channels_last"))
model.add(Convolution2D(64, (2, 3), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform',data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None))
model.add(ZeroPadding2D((0, 2),data_format="channels_last"))
model.add(Convolution2D(32, (1, 3), padding="valid", activation="relu", name="conv3", kernel_initializer='glorot_uniform',data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None))
model.add(Reshape((17,32),input_shape=(1,17,32)))
model.add(LSTM(100,activation='relu', recurrent_activation='hard_sigmoid', use_bias=True,
               kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', return_sequences=True))
model.add(LSTM(50,activation='relu', recurrent_activation='hard_sigmoid', use_bias=True,
               kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', return_sequences=True))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense( len(classes), kernel_initializer='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
# Set up some params
nb_epoch = 100     # number of epochs to train on
batch_size = 512  # training batch size
# perform training ...
#   - call the main training loop in keras for our network+dataset
current_date = datetime.datetime.now().strftime("%-m-%-d")
save_dir = os.path.join(root_dir,"runs/weights/AF-AMC",current_date)
if os.path.exists(save_dir):
    print(f"目录已存在：{save_dir}（将直接使用现有目录）")
else:
    try:
        os.makedirs(save_dir, exist_ok=False)  # 显式关闭exist_ok以触发异常
        print(f"目录创建成功：{save_dir}")
    except FileExistsError:
        print(f"并发冲突：其他进程已创建该目录")
    except PermissionError:
        print(f"权限不足：无法创建目录 {save_dir}")
        raise
filepath = os.path.join(save_dir,"weights.wts.h5")
history = model.fit(X_train,
        Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        #   show_accuracy=False,
        verbose=2,
        validation_data=(X_test, Y_test),
        callbacks = [
        #   如果在每次迭代中发现损失优化的情况则保存模型权重，当五次迭代都没有改进时停止训练
            tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])
# we re-load the best weights once training is finished，即训练完成后加载最佳训练参数
print("训练已经完成，最佳权重模型已保存到根目录！正在加载最佳权重模型....")
model.load_weights(filepath)
# with open('history.json', 'w') as f:
#     json.dump(history.history, f)
#     json.dump(history.epoch, f)
#
print("最佳模型加载成功！")
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("Loss: ", score[0])
print("Accuracy: ", score[1])
#   画出损失曲线
plt.figure()
plt.title('Training performance')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.grid(True)
plt.legend()
path_to_save = f"{save_dir}/loss"  # 替换为你想要保存图像的路径
plt.savefig(path_to_save, format='png', dpi=300)  # 设置 dpi 参数以调整保存的图像质量

#   定义混淆矩阵，输入为矩阵数据，标题，配色和标签，实际调用时只需要提供混淆矩阵数据和标签列表即可。
def plot_confusion_matrix(cm, title='', cmap=plt.cm.Blues, labels=[]):
    #plt.figure(figsize=(10, 8))  # 调整图片尺寸
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    axis_font = {'fontname': 'Times New Roman', 'size': 14}
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=12, fontname='Times New Roman')
    plt.yticks(tick_marks, labels, fontsize=12, fontname='Times New Roman')
    # 对每个位置进行归一化处理并显示归一化后的准确率
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j] > 0.1:
                plt.text(j, i, "{:.2f}".format(cm[i, j]), horizontalalignment="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12, fontname='Times New Roman')


    plt.tight_layout(pad=2.0)  # 增加pad值以提供更多的空间
    plt.subplots_adjust(bottom=0.2)  # 手动调整底部空间
    plt.ylabel('True label', **axis_font)
    plt.xlabel('Predicted label', **axis_font)
    plt.grid(True)

#   批处理对测试集进行预测
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
#   遍历所有测试样本
for i in range(0,X_test.shape[0]):
    #   找到第i个测试样本的标签，因为是二进制编码格式。所以找到的是1的索引
    j = list(Y_test[i,:]).index(1)
    #   找到预测结果中概率最大的元素索引
    k = int(np.argmax(test_Y_hat[i,:]))
    #   对应混淆矩阵上的点加1
    conf[j,k] = conf[j,k] + 1
#   遍历所有类别
for i in range(0,len(classes)):
    #   对原始混淆矩阵进行归一化
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)
plt.savefig(f"{save_dir}/confu_matrix_total.png",format='png', dpi=300,fontname='Times New Roman', bbox_inches='tight')
acc = {}
#   取出测试集的所有信噪比列表，根据测试集索引test_idx来取
test_SNRs = list(map(lambda x: lbl[x][1], test_idx))

#   把指定信噪比下的测试数据提取出来
for snr in snrs:
    #   作判断是不是指定信噪比
    snr_bool = np.array(test_SNRs) == snr
    #   找到匹配的信噪比索引
    snr_idx = np.where(snr_bool)
    #   取出该信噪比下的数据
    test_X_i = X_test[snr_idx]
    #   取出标签
    test_Y_i = Y_test[snr_idx]
    print(len(snr_idx[0]))
    # estimate classes对该信噪比的测试集数据进行预测
    test_Y_i_hat = model.predict(test_X_i, batch_size=batch_size)
    #   初始化混淆矩阵
    conf1 = np.zeros([len(classes), len(classes)])
    confnorm1 = np.zeros([len(classes), len(classes)])
    #   遍历测试样本，构建原始混淆矩阵
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf1[j,k] = conf1[j,k] + 1
    #   归一化混淆矩阵
    for i in range(0,len(classes)):
        confnorm1[i,:] = conf1[i,:] / np.sum(conf1[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm1, labels=classes)
    plt.savefig(f"{save_dir}/comf_Matrix_for_snr=" + str(snr) +".png", format='png', dpi=300,fontname='Times New Roman', bbox_inches='tight')  # 设置 dpi 参数以调整保存的图像质量
    #   拿到原始混淆矩阵对角线的元素并求和
    cor = np.sum(np.diag(conf1))
    #   求出除了对角线元素外的所有元素的和
    ncor = np.sum(conf1) - cor
    #   总体准确率为预测对的数量比上总数
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
# Save results to a pickle file for plotting later
print(acc)
fd = open('./CNN_LSTM/acc_trend.dat', 'wb')
cPickle.dump( ("CNN2", 0.5, acc) , fd )
 # Plot accuracy curve
plt.figure()
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN Classification Accuracy on " + dataFile)
plt.grid(True)
plt.savefig(f"{save_dir}/acc_trend.png", format='png', dpi=1200,fontname='Times New Roman', bbox_inches='tight')  # 设置 dpi 参数以调整保存的图像质量

acc_AFECNN = {-20: 0.6169097538742023, -18: 0.5534747292418772, -16: 0.5915461121157324, -14: 0.5789003129190881, -12: 0.6652331371661385, -10: 0.7345406360424028, -8: 0.8006482982171799, -6: 0.9077770130763937, -4: 0.9511570968484012, -2: 0.9794998873620184, 0: 0.9793416572077185, 2: 0.983097304705345, 4: 0.9769585253456221, 6: 0.989658273381295, 8: 0.9868181818181818, 10: 0.9882973841211565, 12: 0.9933925723399407, 14: 0.9923928077455049, 16: 0.9863945578231292, 18: 0.9826227871084884}
acc_A = {-20: 0.6264813126709207, -18: 0.5742328519855595, -16: 0.6021699819168174, -14: 0.5154224407688869, -12: 0.6482571299230421, -10: 0.6468639575971732, -8: 0.578374623755499, -6: 0.6067905482908924, -4: 0.5189095928226363, -2: 0.6046406848389276, 0: 0.5668558456299659, 2: 0.6187756966651439, 4: 0.5453917050691244, 6: 0.6056654676258992, 8: 0.6056818181818182, 10: 0.5773290500229463, 12: 0.6210982000455685, 14: 0.598893499308437, 16: 0.6229024943310658, 18: 0.56659555152065366}



plt.figure()
plt.yticks(np.arange(0, 1.01, 0.05))
plt.ylim([0, 1.0])  # 设置 y 轴的限制从 0 开始
plt.title('Classification Accuracy')
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.plot(snrs, list(map(lambda x: acc_AFECNN[x], snrs)),label='AFECNN')
plt.plot(snrs, list(map(lambda x: acc_A[x], snrs)),label='artificial features channel')
plt.plot(snrs, list(map(lambda x: acc[x], snrs)),label='adaptive features channel')
plt.grid(True)
plt.legend()
path_to_save = f"{save_dir}/loss.png"  # 替换为你想要保存图像的路径
plt.savefig(path_to_save, format='png', dpi=1200)  # 设置 dpi 参数以调整保存的图像质量