import tensorflow as tf
from tensorflow.keras import layers
import os, time, datetime
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"] = "device=cuda,floatX=float32"
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pickle as cPickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from models.utils.mix_moment.mixd_moment import mixed_moment,self_moments
from models.utils.transformer.MultiHeadAttentionCustom import MultiHeadAttention, TransformerEncoder
times_new_roman_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_manager.fontManager.addfont(times_new_roman_path)
root_dir = "/home/baolin/PycharmProjects/AFECNN"
plt.rcParams['font.family'] = 'Times New Roman'
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
classes[7] = '16QAM'
classes[8] = '64QAM'
#AP数据提取，作为人工特征
Param_R = np.zeros(n_examples)
for i in range(n_examples):
    # 从I/Q数据计算幅度和相位
    I_data = X[i][0]  # 同相分量
    Q_data = X[i][1]  # 正交分量

    # 计算幅度（模长）
    amplitude = np.sqrt(I_data ** 2 + Q_data ** 2)

    # 计算最大/最小幅度比作为特征
    if np.min(amplitude) > 0:
        Param_R[i] = np.max(amplitude) / np.min(amplitude)
    else:
        Param_R[i] = 1.0  # 如果最小幅度为零，则使用默认值
# 以1到8阶矩作为特征参数
# 先计算出复随机变量的混合矩
comp_data_all = []
comp_data = np.zeros(X[0].shape[1])
mom_10 = []
mom_20 = []
mom_21 = []
mom_30 = []
mom_40 = []
mom_41 = []
mom_42 = []
mom_50 = []
mom_60 = []
mom_63 = []
mom_70 = []
mom_80 = []
for i in range(n_examples):
    comp_data = X[i][0] + 1j * X[i][1]
    mom_10.append(self_moments(comp_data,[1]))
    comp_data_all.append(comp_data)
    mom_20.append(self_moments(comp_data,[2]))
    mom_21.append(mixed_moment(comp_data,np.conj(comp_data),1,1))
    mom_30.append(self_moments(comp_data,[3]))
    mom_40.append(self_moments(comp_data,[4]))
    mom_41.append(mixed_moment(comp_data,np.conj(comp_data),3,1))
    mom_42.append(mixed_moment(comp_data,np.conj(comp_data),2,2))
    mom_50.append(self_moments(comp_data,[5]))
    mom_60.append(self_moments(comp_data,[6]))
    mom_63.append(mixed_moment(comp_data,np.conj(comp_data),3,3))
    mom_70.append(self_moments(comp_data,[7]))
    mom_80.append(self_moments(comp_data,[8]))
# 以2到8阶累积量构造特征参数
#  计算高阶累积量，在这个列表推导式中，我们使用了 zip 函数，该函数用于将两个可迭代对象逐一配对，形成一个元组。
C_20 = mom_20
C_21 = mom_21
C_40 = [mom_40 - 3 * mom_20**2 for mom_40, mom_20 in zip(mom_40, mom_20)]
C_41 = [mom_41 - 3 * mom_20*mom_21 for mom_41, mom_20,mom_21 in zip(mom_41, mom_20,mom_21)]
C_42 = [mom_42 - mom_20**2 - 2*mom_21**2 for mom_42, mom_20, mom_21 in zip(mom_42, mom_20, mom_21)]
C_60 = [mom_60 - 15*mom_40*mom_20 + 30*mom_20**3 for mom_60, mom_40, mom_20 in zip(mom_60, mom_40, mom_20)]
C_63 = [mom_63 - 9*C_42*C_21 - 6*C_21**3 for mom_63, C_42, C_21 in zip(mom_63, C_42, C_21)]
C_80 = [mom_80 - 28*mom_60*mom_20 - 35*mom_40**2 + 420*mom_40*mom_20**2 - 630*mom_20**4 for mom_80, mom_60,mom_20,mom_40
        in zip(mom_80, mom_60,mom_20,mom_40)]
# 构建特征参数
M_1 = [(c20 / c21) for c20, c21 in zip(C_20, C_21)]
M_3 = [(C_40 / C_42) for C_40,C_42 in zip(C_40,C_42)]
M_2 = [np.abs(c42 / c21**2) for c42, c21 in zip(C_42,C_21)]
M_4 = [np.abs(c40 / c21**2) for c40, c21 in zip(C_40,C_21)]
M_5 = [np.abs(c63 / c21**3) for c63, c21 in zip(C_63,C_21)]
M_6 = [np.abs(c80 / c21**4) for c80, c21 in zip(C_80,C_21)]
extraData = [Param_R,np.real(M_1),np.imag(M_1),M_2,
             np.real(M_3),np.imag(M_3),M_4,M_5,M_6,np.real(C_60),np.imag(C_60)]

# 将子列表转换成np数组利于转换为二维
extraData_same = [np.array(item).flatten() for item in extraData]
# 将所有子数组堆叠为一个二维数组
extraData_np = np.array(extraData_same).reshape((len(extraData_same[0]),len(extraData_same)))
extraData_train = np.real(extraData_np[train_idx])
extraData_test = np.real(extraData_np[test_idx])
# 将 NumPy 数组转换为 TensorFlow 张量
extraData_train_tf = tf.convert_to_tensor(extraData_train, dtype=tf.float32)
extraData_test_tf = tf.convert_to_tensor(extraData_test, dtype=tf.float32)
in_shp_2 = list(extraData_train.shape[1:])
print(extraData_train.shape,in_shp_2)
extra_Param = tf.keras.layers.Input(shape=(in_shp_2))

def get_positional_encoding(sequence_length, d_model):
    angle_rads = np.arange(sequence_length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :]//2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


#第二层
dr = 0.5
input_2 = tf.keras.layers.Input(shape=in_shp)
b = tf.keras.layers.Reshape(in_shp+[1])(input_2)
b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
b = tf.keras.layers.Convolution2D(128, (1, 3),padding='valid', activation="relu", name="conv4", kernel_initializer='glorot_uniform', data_format="channels_last")(b)
b = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(b)
b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(b)
b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
b = tf.keras.layers.Convolution2D(64, (2, 3), padding='valid', activation="relu", name="conv5", kernel_initializer='glorot_uniform', data_format="channels_last")(b)
b = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(b)
b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(b)
b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
b = tf.keras.layers.Convolution2D(32, (1, 3), padding='valid', activation="relu", name="conv6", kernel_initializer='glorot_uniform', data_format="channels_last")(b)
b = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(b)
b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(b)
b = tf.keras.layers.Reshape((17, 32),input_shape=(1, 17, 32))(b)

# 添加位置编码
pos_encoding = get_positional_encoding(sequence_length=17, d_model=32)
b = layers.Add()([b, pos_encoding])

# Transformer Encoder层
transformer_block = TransformerEncoder(num_heads=4, key_dim=32, ff_dim=64, dropout=0.1)
b = transformer_block(b, training=True)  # (batch_size, seq_len, key_dim)
b = transformer_block(b, training=True)  # 可堆叠多个TransformerEncoder块
b = transformer_block(b, training=True)  # 可堆叠多个TransformerEncoder块
output_2 = layers.GlobalAveragePooling1D(name="global_average_pooling1d")(b)
# output_2 = tf.keras.layers.Flatten()(b)
# # 混合池化
# max_pool = layers.GlobalMaxPooling1D()(b)
# avg_pool = layers.GlobalAveragePooling1D()(b)
# output_2 = layers.concatenate([max_pool, avg_pool])  # (batch_size, key_dim)
second = tf.keras.Model(input_2, output_2, name="second")
second.summary()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

class WeightedFusion(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedFusion, self).__init__(**kwargs)
        self.alpha = tf.Variable(0.5, trainable=True, dtype=tf.float32, name='fusion_alpha')

    def call(self, inputs):
        x1, x2 = inputs
        return self.alpha * x1 + (1.0 - self.alpha) * x2



# 分别映射到统一维度
output_1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_1")(extra_Param)
output_2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_2")(output_2)
# 加权融合
fused = WeightedFusion()([output_1, output_2])
fused = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_3")(fused)
fused = tf.keras.layers.Dropout(dr)(fused)
fused = tf.keras.layers.Dense(len(classes), kernel_initializer='he_normal', name='dense_4')(fused)
fused = tf.keras.layers.Activation('softmax')(fused)
output = tf.keras.layers.Reshape([len(classes)])(fused)
final = tf.keras.Model(inputs=[extra_Param, input_2], outputs=output)
final.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
final.summary()


# 设置一些参数
nb_epoch = 100  # number of epochs to train on
batch_size = 512  # training batch size
current_date = datetime.datetime.now().strftime("%-m-%-d-%H-%M")
save_dir = os.path.join(root_dir,"runs/AFECNN",current_date)
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
filepath3 = os.path.join(save_dir,"weights.wts.h5")
# Set up some params
history3 = final.fit([extraData_train_tf, X_train],                         #训练数据
                    Y_train,                         #训练数据对应的标签
                    batch_size=batch_size,           #训练批量大小，每次训练时使用的样本数
                    epochs=nb_epoch,                 #训练轮数，表示模型需要训练的次数
                    #show_accuracy=False,            #表示不显示训练过程中的准确率
                    verbose=2,                       #表示在训练过程中显示详细信息
                    validation_data=([extraData_test_tf, X_test],Y_test), #使用测试数据X_test和标签Y_text进行模型的验证
                    callbacks = [                    #回调函数，用于在每个训练轮数结束时保存模型的权重
                        tf.keras.callbacks.ModelCheckpoint(filepath3, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),#filepath是保存模型权重的路径
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')#回调函数，用于在验证集上损失函数不再下降时停止训练，5次迭代都没有改进时停止训练
                    ])
print ("训练已经完成，最佳权重模型已保存到根目录！正在加载最佳权重模型....")
# final.load_weights(filepath3)
final.load_weights(filepath3)
print ("最佳模型加载成功！")
score = final.evaluate([extraData_test_tf, X_test], Y_test, verbose=0, batch_size=batch_size)
print("Loss: ", score[0])
print("Accuracy: ", score[1])

#   画出损失曲线
plt.figure()
plt.yticks(np.arange(0, 5.0, 0.5))
plt.ylim([0, 5.0])  # 设置 y 轴的限制从 0 开始
plt.title('Training performance')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history3.epoch, history3.history['loss'], label='train loss+error')
plt.plot(history3.epoch, history3.history['val_loss'], label='val_error')
plt.grid(True)
plt.legend()
plt.savefig(f"{save_dir}/loss.png", format='png', dpi=1200,bbox_inches='tight')





def feature_correlation_analysis(extraData_train, classes, train_idx, lbl):
    """Analyze correlations between features and modulation types"""
    # Create DataFrame with features and modulation types
    feature_names = ['Param_R', 'M_1_real', 'M_1_imag', 'M_2', 'M_3_real', 'M_3_imag',
                     'M_4', 'M_5', 'M_6', 'C_60_real', 'C_60_imag']
    df = pd.DataFrame(extraData_train, columns=feature_names)
    df['modulation'] = [lbl[i][0] for i in train_idx]

    # Calculate correlation matrix
    corr_matrix = df[feature_names].corr()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')

    # Calculate feature-modulation correlations
    mod_correlations = {}
    for mod in classes:
        mod_mask = df['modulation'] == mod
        mod_correlations[mod] = df[feature_names][mod_mask].mean()

    return corr_matrix, mod_correlations


def pca_analysis(extraData_train, classes, train_idx, lbl):
    """Perform PCA analysis on artificial features"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(extraData_train)

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1),
             cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig('pca_variance.png')

    # Plot first two principal components
    plt.figure(figsize=(12, 8))
    class_qam = ['QAM16', 'QAM64', 'WBFM']
    for mod in class_qam:
        mod_mask = [lbl[i][0] == mod for i in train_idx]
        plt.scatter(X_pca[mod_mask, 0], X_pca[mod_mask, 1], label=mod,s=5)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of Artificial Features')
    plt.legend()
    plt.grid(True)
    plt.savefig('pca_scatter.png')

    return pca, X_pca

# 人工特征相关性分析
feature_correlation_analysis(extraData_train, classes, train_idx, lbl)
pca_analysis(extraData_train, classes, train_idx, lbl)

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
start_time = time.time()
test_Y_hat = final.predict([extraData_test_tf,X_test], batch_size=batch_size)
end_time = time.time()
print(start_time-end_time)
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

snr_accuracy = {snr: {mod: 0 for mod in classes} for snr in snrs}

#   把指定信噪比下的测试数据提取出来
for snr in snrs:
    #   作判断是不是指定信噪比
    snr_bool = np.array(test_SNRs) == snr
    #   找到匹配的信噪比索引
    snr_idx = np.where(snr_bool)
    #   取出该信噪比下的数据
    test_X_i = X_test[snr_idx]
    test_M_i = extraData_test[snr_idx]
    test_M_i = tf.convert_to_tensor(test_M_i, dtype=tf.float32)
    # test_M_i = np.reshape(test_M_i, (5456, 5))
    #   取出标签
    test_Y_i = Y_test[snr_idx]
    print(len(snr_idx[0]))
    # estimate classes对该信噪比的测试集数据进行预测
    test_Y_i_hat = final.predict([test_M_i,test_X_i], batch_size=batch_size)
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
    # 计算每种调制方式的准确率并存入字典
    for i, mod in enumerate(classes):
        snr_accuracy[snr][mod] = confnorm1[i, i]
    plt.figure()
    plot_confusion_matrix(confnorm1, labels=classes)
    plt.savefig(f"{save_dir}/comf_Matrix_for_snr=" + str(snr)+".png", format='png', dpi=300, bbox_inches='tight')  # 设置 dpi 参数以调整保存的图像质量
    #   拿到原始混淆矩阵对角线的元素并求和
    cor = np.sum(np.diag(conf1))
    #   求出除了对角线元素外的所有元素的和
    ncor = np.sum(conf1) - cor
    #   总体准确率为预测对的数量比上总数
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
# Save results to a pickle file for plotting later
print(acc)
with open(f"{save_dir}/acc_results.txt", "w") as f:
    f.write(str(acc))
acc_CNN = {-20:0.09,-18:0.09,-16:0.09,-14:0.1,-12:0.13,-10:0.21,-8:0.325,-6:0.469,
           -4:0.57,-2:0.68,0:0.73,2:0.76,4:0.775,6:0.75,8:0.775,10:0.785,12:0.775,14:0.77,16:0.762,18:0.78}
acc_CGDNN = {-20:0.09,-18:0.09,-16:0.09,-14:0.1,-12:0.13,-10:0.21,-8:0.325,-6:0.5,
           -4:0.67,-2:0.72,0:0.76,2:0.775,4:0.8,6:0.78,8:0.8,10:0.82,12:0.8,14:0.78,16:0.79,18:0.81}
acc_IC_AMCNet = {-20:0.09,-18:0.09,-16:0.09,-14:0.1,-12:0.13,-10:0.21,-8:0.35,-6:0.51,
           -4:0.61,-2:0.7,0:0.78,2:0.8,4:0.815,6:0.825,8:0.825,10:0.84,12:0.83,14:0.822,16:0.82,18:0.836}
acc_CLDNN = {-20:0.09,-18:0.09,-16:0.09,-14:0.1,-12:0.13,-10:0.195,-8:0.35,-6:0.551,
           -4:0.685,-2:0.78,0:0.81,2:0.815,4:0.83,6:0.81,8:0.83,10:0.838,12:0.82,14:0.823,16:0.821,18:0.836}
acc_CLDNN2 = {-20:0.09,-18:0.09,-16:0.09,-14:0.13,-12:0.17,-10:0.24,-8:0.37,-6:0.56,
           -4:0.67,-2:0.80,0:0.835,2:0.86,4:0.878,6:0.866,8:0.88,10:0.9,12:0.91,14:0.882,16:0.888,18:0.88}
acc_PET_CGDNN = {-20:0.09,-18:0.09,-16:0.1,-14:0.13,-12:0.175,-10:0.252,-8:0.38,-6:0.525,
           -4:0.635,-2:0.775,0:0.858,2:0.881,4:0.916,6:0.90,8:0.916,10:0.912,12:0.92,14:0.905,16:0.903,18:0.918}
acc_MCLDNN = {-20:0.09,-18:0.1,-16:0.11,-14:0.13,-12:0.18,-10:0.275,-8:0.42,-6:0.55,
           -4:0.67,-2:0.79,0:0.88,2:0.898,4:0.922,6:0.91,8:0.92,10:0.92,12:0.918,14:0.919,16:0.915,18:0.918}
acc_MCL_BigNetNN = {-20:0.1,-18:0.12,-16:0.12,-14:0.15,-12:0.17,-10:0.28,-8:0.472,-6:0.58,
           -4:0.68,-2:0.852,0:0.893,2:0.915,4:0.92,6:0.921,8:0.922,10:0.94,12:0.938,14:0.945,16:0.939,18:0.933}
with open(f"{root_dir}/runs/AFECNN/acc_trend.dat", 'wb') as file:
    cPickle.dump(acc, file)



plt.figure()
plt.yticks(np.arange(0, 1.01, 0.05))
plt.ylim([0, 1.0])  # 设置 y 轴的限制从 0 开始
plt.plot(snrs, list(map(lambda x: acc[x], snrs)),
            marker='o',
            markersize=4,
            markerfacecolor='red',
            linestyle='-',
            color='red',
            label='AFECNN_TRN')
plt.plot(snrs, list(map(lambda x: acc_CNN[x], snrs)),
            marker='s',
            markersize=4,
            markerfacecolor='green',
            linestyle='-',
            color='green',
            label='CNN')
plt.plot(snrs, list(map(lambda x: acc_CGDNN[x], snrs)),
            marker='^',
            markersize=4,
            markerfacecolor='lightgreen',
            linestyle='-',
            color='lightgreen',
            label='CGDNN')
plt.plot(snrs, list(map(lambda x: acc_IC_AMCNet[x], snrs)),
            marker='*',
            markersize=4,
            markerfacecolor='tomato',
            linestyle='-',
            color='tomato',
            label='IC_AMCNet')
plt.plot(snrs, list(map(lambda x: acc_CLDNN[x], snrs)),
            marker='+',
            markersize=4,
            markerfacecolor='blue',
            linestyle='-',
            color='blue',
            label='CLDNN')
plt.plot(snrs, list(map(lambda x: acc_CLDNN2[x], snrs)),
            marker='x',
            markersize=4,
            markerfacecolor='hotpink',
            linestyle='-',
            color='hotpink',
            label='CLDNN2')
plt.plot(snrs, list(map(lambda x: acc_PET_CGDNN[x], snrs)),
            marker='>',
            markersize=4,
            markerfacecolor='dodgerblue',
            linestyle='-',
            color='dodgerblue',
            label='PET_CGDNN')
plt.plot(snrs, list(map(lambda x: acc_MCLDNN[x], snrs)),
            marker='o',
            markersize=4,
            markerfacecolor='darkblue',
            linestyle='-',
            color='darkblue',
            label='MCLDNN')
plt.plot(snrs, list(map(lambda x: acc_MCL_BigNetNN[x], snrs)),
            marker='<',
            markersize=4,
            markerfacecolor='orange',
            linestyle='-',
            color='orange',
            label='MCL_BigRU3NetNN')
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("")
plt.grid(True)
plt.legend(loc='upper left')
ax_sub = inset_axes(plt.gca(), width='40%', height='40%', loc='center right')

# 在子图上绘制相同的数据或者是数据的一个子集
ax_sub.plot(snrs, list(map(lambda x: acc[x], snrs)),
            marker='o',
            markersize=2,
            markerfacecolor='red',
            linestyle='-',
            color='red')
ax_sub.plot(snrs, list(map(lambda x: acc_CNN[x], snrs)),
            marker='s',
            markersize=2,
            markerfacecolor='green',
            linestyle='-',
            color='green',)
ax_sub.plot(snrs, list(map(lambda x: acc_CGDNN[x], snrs)),
            marker='^',
            markersize=2,
            markerfacecolor='lightgreen',
            linestyle='-',
            color='lightgreen',
            label='CGDNN')
ax_sub.plot(snrs, list(map(lambda x: acc_IC_AMCNet[x], snrs)),
            marker='*',
            markersize=2,
            markerfacecolor='tomato',
            linestyle='-',
            color='tomato',
            label='IC_AMCNet')
ax_sub.plot(snrs, list(map(lambda x: acc_CLDNN[x], snrs)),
            marker='+',
            markersize=2,
            markerfacecolor='blue',
            linestyle='-',
            color='blue',
            label='CLDNN')
ax_sub.plot(snrs, list(map(lambda x: acc_CLDNN2[x], snrs)),
            marker='x',
            markersize=2,
            markerfacecolor='hotpink',
            linestyle='-',
            color='hotpink',
            label='CLDNN2')
ax_sub.plot(snrs, list(map(lambda x: acc_PET_CGDNN[x], snrs)),
            marker='>',
            markersize=2,
            markerfacecolor='dodgerblue',
            linestyle='-',
            color='dodgerblue',
            label='PET_CGDNN')
ax_sub.plot(snrs, list(map(lambda x: acc_MCLDNN[x], snrs)),
            marker='o',
            markersize=2,
            markerfacecolor='darkblue',
            linestyle='-',
            color='darkblue',
            label='MCLDNN')
ax_sub.plot(snrs, list(map(lambda x: acc_MCL_BigNetNN[x], snrs)),
            marker='<',
            markersize=2,
            markerfacecolor='orange',
            linestyle='-',
            color='orange',
            label='MCL_BigNetNN')
# 可以设置子图的坐标轴标签和标题
ax_sub.set_xlabel('SNR (dB)', fontsize=8)
ax_sub.set_ylabel('', fontsize=8)
# 设置子图的 x 轴和 y 轴的限制，以放大感兴趣的部分
ax_sub.set_xlim([0, 18])
ax_sub.set_ylim([0.85, 1.0])
ax_sub.grid(True)
# 设置子图的刻度标签大小
ax_sub.tick_params(labelsize=8)
plt.savefig(f"{save_dir}/acc_trend.png", format='png', dpi=1200)  # 设置 dpi 参数以调整保存的图像质量


plt.figure()
plt.yticks(np.arange(0, 1.01, 0.05))
plt.ylim([0, 1.0])  # 设置 y 轴的限制从 0 开始
for mod in classes:
    plt.plot(snrs, [snr_accuracy[snr][mod] for snr in snrs], marker='o', label=mod)
plt.xlabel('Signal to Noise Ratio (SNR)')
plt.ylabel('Classification Accuracy')
plt.title('')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig(f"{save_dir}/acc_trend_for_all_styles.png", format='png', dpi=1200)  # 设置 dpi 参数以调整保存的图像质量

acc={-20: 0.6424339106654512, -18: 0.5304602888086642, -16: 0.6069168173598554, -14: 0.5822530174340634, -12: 0.7385694884563151, -10: 0.7948321554770318, -8: 0.8159180365825422, -6: 0.8839183298921771, -4: 0.94, -2: 0.9774724036945258, 0: 0.9822928490351873, 2: 0.9828688899040657, 4: 0.976036866359447, 6: 0.9923561151079137, 8: 0.9911363636363636, 10: 0.9866911427260211, 12: 0.9915698336750969, 14: 0.9905486399262333, 16: 0.9861678004535147, 18: 0.976395823876532}
acc_adaptive_weight={-20: 0.6490428441203282, -18: 0.5625, -16: 0.6182188065099458, -14: 0.6097451944568618, -12: 0.6944318696242644, -10: 0.778047703180212, -8: 0.8196341745774485, -6: 0.9091534755677908, -4: 0.954803312629399, -2: 0.980851543140347, 0: 0.9822928490351873, 2: 0.9865235267245317, 4: 0.9866359447004608, 6: 0.9921312949640287, 8: 0.990909090909091, 10: 0.9919687930243231, 12: 0.9922533606744133, 14: 0.9905486399262333, 16: 0.9859410430839002, 18: 0.980708125283704}
acc_gate={-20: 0.6679580674567, -18: 0.5852888086642599, -16: 0.6503164556962026, -14: 0.7221725525257041, -12: 0.7505658669081032, -10: 0.814708480565371, -8: 0.8430192174114378, -6: 0.9208534067446662, -4: 0.9641936968023925, -2: 0.977247127731471, 0: 0.9763904653802498, 2: 0.9787574234810416, 4: 0.9824884792626728, 6: 0.989658273381295, 8: 0.9895454545454545, 10: 0.9899036255162919, 12: 0.988835725677831, 14: 0.9859382203780545, 16: 0.9854875283446712, 18: 0.9788924194280526}
plt.figure()
plt.yticks(np.arange(0, 1.01, 0.05))
plt.ylim([0, 1.0])  # 设置 y 轴的限制从 0 开始
plt.plot(snrs, list(map(lambda x: acc[x], snrs)),
            marker='o',
            markersize=4,
            markerfacecolor='red',
            linestyle='-',
            color='red',
            label='simple_concat')
plt.plot(snrs, list(map(lambda x: acc_adaptive_weight[x], snrs)),
            marker='s',
            markersize=4,
            markerfacecolor='green',
            linestyle='-',
            color='green',
            label='Attention')
plt.plot(snrs, list(map(lambda x: acc_gate[x], snrs)),
            marker='^',
            markersize=4,
            markerfacecolor='orange',
            linestyle='-',
            color='orange',
            label='Gate')
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("")
plt.grid(True)
plt.legend(loc='upper left')
ax_sub = inset_axes(plt.gca(), width='40%', height='40%', loc='center right')

# 在子图上绘制相同的数据或者是数据的一个子集

ax_sub.plot(snrs, list(map(lambda x: acc[x], snrs)),
            marker='o',
            markersize=2,
            markerfacecolor='red',
            linestyle='-',
            color='red',
            label='simple_concat')
ax_sub.plot(snrs, list(map(lambda x: acc_adaptive_weight[x], snrs)),
            marker='s',
            markersize=2,
            markerfacecolor='green',
            linestyle='-',
            color='green',
            label='Attention')
ax_sub.plot(snrs, list(map(lambda x: acc_gate[x], snrs)),
            marker='^',
            markersize=2,
            markerfacecolor='orange',
            linestyle='-',
            color='orange',
            label='Gate')
# 可以设置子图的坐标轴标签和标题
ax_sub.set_xlabel('SNR (dB)', fontsize=8)
ax_sub.set_ylabel('', fontsize=8)
# 设置子图的 x 轴和 y 轴的限制，以放大感兴趣的部分
ax_sub.set_xlim([0, 18])
ax_sub.set_ylim([0.85, 1.0])
ax_sub.grid(True)
# 设置子图的刻度标签大小
ax_sub.tick_params(labelsize=8)
plt.savefig("acc_trend_3.png", format='png', dpi=1200)  # 设置 dpi 参数以调整保存的图像质量
