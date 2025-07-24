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
# extraData = [Param_R,np.real(M_1),np.imag(M_1),M_2,
#              np.real(M_3),np.imag(M_3),M_4,M_5,M_6,np.real(C_60),np.imag(C_60)]
extraData = [Param_R,np.real(M_1),np.imag(M_1),M_2
            ,np.real(M_3),np.imag(M_3),M_4,M_5,M_6,np.real(C_60)]
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
b = tf.keras.layers.Convolution2D(64, (1, 3),padding='valid', activation="relu", name="conv4", kernel_initializer='glorot_uniform', data_format="channels_last")(b)
b = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(b)
b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(b)
b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
b = tf.keras.layers.Convolution2D(32, (2, 3), padding='valid', activation="relu", name="conv5", kernel_initializer='glorot_uniform', data_format="channels_last")(b)
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



# 分别映射到统一维度
output_1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_1")(extra_Param)
output_2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_2")(output_2)
# 加权融合
concat = tf.keras.layers.Concatenate(axis=-1)([output_2,output_1])
# 门控向量：大小与 deep_features 相同
gate = tf.keras.layers.Dense(units=output_2.shape[-1], activation='sigmoid')(concat)
fused = tf.keras.layers.Multiply()([gate, output_2])
inv_gate = tf.keras.layers.Lambda(lambda x: 1.0 - x)(gate)
hand_weighted = tf.keras.layers.Multiply()([inv_gate, output_1])
fused = tf.keras.layers.Add()([fused, hand_weighted])
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
with open(f"{save_dir}/acc_results.txt", "w") as f:
    f.write("Loss:" + str(score[0]) + "\n" + "Accuracy:" + str(score[1]) + "\n" + str(final.summary()))

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
    feature_names = ['M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6',
                     'M_7', 'M_8', 'M_9', 'M_10', 'M_11']
    df = pd.DataFrame(extraData_train, columns=feature_names)
    df['modulation'] = [lbl[i][0] for i in train_idx]

    # Calculate correlation matrix
    corr_matrix = df[feature_names].corr()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    axis_font = {'fontname': 'Times New Roman', 'size': 14}
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_correlation.png", format='png', dpi=1200, fontname='Times New Roman',
                bbox_inches='tight')
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
    axis_font = {'fontname': 'Times New Roman', 'size': 14}
    plt.plot(range(1, len(explained_variance_ratio) + 1),
             cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components',**axis_font)
    plt.ylabel('Cumulative Explained Variance Ratio',**axis_font)
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig(f"{save_dir}/pca_variance.png", format='png', dpi=1200, fontname='Times New Roman',
                bbox_inches='tight')

    # Plot first two principal components
    plt.figure(figsize=(12, 8))
    axis_font = {'fontname': 'Times New Roman', 'size': 14}
    class_qam = ['QAM16', 'QAM64', 'WBFM']
    for mod in class_qam:
        mod_mask = [lbl[i][0] == mod for i in train_idx]
        plt.scatter(X_pca[mod_mask, 0], X_pca[mod_mask, 1], label=mod,s=5)
    plt.xlabel('First Principal Component',**axis_font)
    plt.ylabel('Second Principal Component',**axis_font)
    plt.title('PCA of Artificial Features')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/pca_scatter.png", format='png', dpi=1200, fontname='Times New Roman',
                bbox_inches='tight')
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
plt.savefig(f"{save_dir}/confu_matrix_total.png",format='png', dpi=1200,fontname='Times New Roman', bbox_inches='tight')
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
    plt.savefig(f"{save_dir}/comf_Matrix_for_snr=" + str(snr)+".png", format='png', dpi=1200, bbox_inches='tight')  # 设置 dpi 参数以调整保存的图像质量
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
    f.write(str(acc) + "\n" + "Loss:" + str(score[0]) + "\n" + "Accuracy:" + str(score[1]) + "\n" + str(final.summary()))

DAE = {-20: 0.09219139999999999, -18: 0.0955434, -16: 0.1000127, -14: 0.12682859999999999, -12: 0.1704045, -10: 0.23856149999999998, -8: 0.3670543, -6: 0.529067, -4: 0.6664983999999999, -2: 0.7927566, 0: 0.8843776, 2: 0.9011374999999999, 4: 0.9167801, 6: 0.9089588000000001, 8: 0.9134281, 10: 0.9178974, 12: 0.9201321, 14: 0.9145454, 16: 0.9111933999999999, 18: 0.9190147000000001}
R_ResNet = {-20: 0.09666079999999999, -18: 0.09666079999999999, -16: 0.1033647, -14: 0.12459400000000001, -12: 0.1715218, -10: 0.2519695, -8: 0.35811570000000004, -6: 0.5011338, -4: 0.6117493, -2: 0.7167781999999999, 0: 0.7625887, 2: 0.7849353000000001, 4: 0.7938738999999999, 6: 0.8106339, 8: 0.8005779, 10: 0.8095165, 12: 0.8050472000000001, 14: 0.8151031999999999, 16: 0.7894046, 18: 0.8095165}
MCLDNN = {-20: 0.0944261, -18: 0.0944261, -16: 0.10671670000000001, -14: 0.121242, -12: 0.1614658, -10: 0.2519695, -8: 0.3871662, -6: 0.5469442999999999, -4: 0.6720851, -2: 0.8117512000000001, 0: 0.8866122, 2: 0.9078415, 4: 0.9145454, 6: 0.9123108000000001, 8: 0.9190147000000001, 10: 0.9212494, 12: 0.9190147000000001, 14: 0.9234841, 16: 0.9145454, 18: 0.9223667}
PET_CGDNN= {-20: 0.0944261, -18: 0.09330880000000001, -16: 0.1033647, -14: 0.1190073, -12: 0.1681698, -10: 0.24303080000000002, -8: 0.3860489, -6: 0.5156591, -4: 0.6318612, -2: 0.767058, 0: 0.8441537, 2: 0.8810256, 4: 0.8977854999999999, 6: 0.8955508000000001, 8: 0.8989028, 10: 0.9056068, 12: 0.9056068, 14: 0.9000202, 16: 0.9011374999999999, 18: 0.9022548}
convLSTMAE = {-20: 0.0944261, -18: 0.09107409999999999, -16: 0.0955434, -14: 0.12347670000000001, -12: 0.16593509999999997, -10: 0.23744420000000002, -8: 0.3871662, -6: 0.5480616, -4: 0.7011356000000001, -2: 0.8095165, 0: 0.8843776, 2: 0.9123108000000001, 4: 0.9201321, 6: 0.9190147000000001, 8: 0.930188, 10: 0.9223667, 12: 0.9190147000000001, 14: 0.9290707, 16: 0.9156628000000001, 18: 0.9279533999999999}
C_CNN ={-20: 0.09666079999999999, -18: 0.0977781, -16: 0.10895139999999999, -14: 0.09666079999999999, -12: 0.1368846, -10: 0.21062830000000002, -8: 0.33018250000000005, -6: 0.4877258, -4: 0.5659388, -2: 0.6910797, 0: 0.773762, 2: 0.7972258999999999, 4: 0.8184551999999999, 6: 0.8139858, 8: 0.8206897999999999, 10: 0.8218071, 12: 0.8184551999999999, 14: 0.8251591, 16: 0.8072819, 18: 0.8128685}
LSTM2 = {-20: 0.0955434, -18: 0.09666079999999999, -16: 0.1044821, -14: 0.1190073, -12: 0.1558792, -10: 0.2318575, -8: 0.3681717, -6: 0.5335363000000001, -4: 0.6687331000000001, -2: 0.7905219, 0: 0.8631483, 2: 0.9011374999999999, 4: 0.9167801, 6: 0.9111933999999999, 8: 0.9156628000000001, 10: 0.9212494, 12: 0.9201321, 14: 0.9145454, 16: 0.9134281, 18: 0.9212494}
C_ResNet = {-20: 0.0955434, -18: 0.0955434, -16: 0.1134207, -14: 0.1201247, -12: 0.1491752, -10: 0.2240362, -8: 0.30895320000000004, -6: 0.40727820000000003, -4: 0.5268323, -2: 0.6553252, 0: 0.7491808, 2: 0.7849353000000001, 4: 0.7871699, 6: 0.8117512000000001, 8: 0.8095165, 10: 0.8162205, 12: 0.8050472000000001, 14: 0.8139858, 16: 0.8016952, 18: 0.8117512000000001}
CGF_HNN = {-20: 0.0944261, -18: 0.09666079999999999, -16: 0.1055994, -14: 0.131298, -12: 0.1726391, -10: 0.26649470000000003, -8: 0.4173341, -6: 0.5961067, -4: 0.7301862, -2: 0.8430364, 0: 0.9100760999999999, 2: 0.9212494, 4: 0.9290707, 6: 0.926836, 8: 0.9290707, 10: 0.9357747000000001, 12: 0.93354, 14: 0.9324227, 16: 0.926836, 18: 0.9346572999999999}


with open(f"{root_dir}/runs/AFECNN/acc_trend.dat", 'wb') as file:
    cPickle.dump(acc, file)


plt.figure()
plt.yticks(np.arange(0, 1.01, 0.05))
plt.ylim([0, 1.0])  # 设置 y 轴的限制从 0 开始
plt.plot(snrs, list(map(lambda x: DAE[x], snrs)),
            marker='<',
            markersize=4,
            markerfacecolor='orange',
            linestyle='-',
            color='orange',
            label='DAE')
plt.plot(snrs, list(map(lambda x: R_ResNet[x], snrs)),
            marker='s',
            markersize=4,
            markerfacecolor='green',
            linestyle='-',
            color='green',
            label='R-ResNet')
plt.plot(snrs, list(map(lambda x: MCLDNN[x], snrs)),
            marker='^',
            markersize=4,
            markerfacecolor='lightgreen',
            linestyle='-',
            color='lightgreen',
            label='MCLDNN')
plt.plot(snrs, list(map(lambda x: PET_CGDNN[x], snrs)),
            marker='*',
            markersize=4,
            markerfacecolor='tomato',
            linestyle='-',
            color='tomato',
            label='PET-CGDNN')
plt.plot(snrs, list(map(lambda x: convLSTMAE[x], snrs)),
            marker='+',
            markersize=4,
            markerfacecolor='blue',
            linestyle='-',
            color='blue',
            label='convLSTMAE')
plt.plot(snrs, list(map(lambda x: CGF_HNN[x], snrs)),
            marker='>',
            markersize=4,
            markerfacecolor='hotpink',
            linestyle='-',
            color='hotpink',
            label='CGF-HNN')
plt.plot(snrs, list(map(lambda x: LSTM2[x], snrs)),
            marker='>',
            markersize=4,
            markerfacecolor='dodgerblue',
            linestyle='-',
            color='dodgerblue',
            label='LSTM2')
plt.plot(snrs, list(map(lambda x: C_ResNet[x], snrs)),
            marker='o',
            markersize=4,
            markerfacecolor='darkblue',
            linestyle='-',
            color='darkblue',
            label='C-ResNet')

plt.plot(snrs, list(map(lambda x: acc, snrs)),
            marker='o',
            markersize=4,
            markerfacecolor='red',
            linestyle='-',
            color='red',
            label='K-CTHN')

plt.xlabel("Signal to Noise Ratio",fontsize=12)
plt.ylabel("Classification Accuracy",fontsize=12)
plt.title("")
plt.grid(True)
plt.legend(loc='upper left',fontsize=10)
ax_sub = inset_axes(plt.gca(), width='40%', height='40%', loc='center right')

# 在子图上绘制相同的数据或者是数据的一个子集
ax_sub.plot(snrs, list(map(lambda x: acc, snrs)),
            marker='o',
            markersize=2,
            markerfacecolor='red',
            linestyle='-',
            color='red',
            label='AFECNN')

ax_sub.plot(snrs, list(map(lambda x: R_ResNet[x], snrs)),
            marker='s',
            markersize=2,
            markerfacecolor='green',
            linestyle='-',
            color='green',
            label='R_ResNet')
ax_sub.plot(snrs, list(map(lambda x: MCLDNN[x], snrs)),
            marker='^',
            markersize=2,
            markerfacecolor='lightgreen',
            linestyle='-',
            color='lightgreen',
            label='MCLDNN')
ax_sub.plot(snrs, list(map(lambda x: PET_CGDNN[x], snrs)),
            marker='*',
            markersize=2,
            markerfacecolor='tomato',
            linestyle='-',
            color='tomato',
            label='PET_CGDNN')
ax_sub.plot(snrs, list(map(lambda x: convLSTMAE[x], snrs)),
            marker='+',
            markersize=2,
            markerfacecolor='blue',
            linestyle='-',
            color='blue',
            label='convLSTMAE')
ax_sub.plot(snrs, list(map(lambda x: CGF_HNN[x], snrs)),
            marker='>',
            markersize=2,
            markerfacecolor='hotpink',
            linestyle='-',
            color='hotpink',
            label='CGF_HNN')
ax_sub.plot(snrs, list(map(lambda x: LSTM2[x], snrs)),
            marker='>',
            markersize=2,
            markerfacecolor='dodgerblue',
            linestyle='-',
            color='dodgerblue',
            label='LSTM2')
ax_sub.plot(snrs, list(map(lambda x: C_ResNet[x], snrs)),
            marker='o',
            markersize=2,
            markerfacecolor='darkblue',
            linestyle='-',
            color='darkblue',
            label='C_ResNet')
ax_sub.plot(snrs, list(map(lambda x: DAE[x], snrs)),
            marker='<',
            markersize=2,
            markerfacecolor='orange',
            linestyle='-',
            color='orange',
            label='DAE')
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
