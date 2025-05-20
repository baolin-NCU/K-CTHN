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
filename = f"{root_dir}/data/A_P_data.pickle"
with open(filename, 'rb') as file:
    M = cPickle.load(file)

# 构建最大半径与最小半径之比为特征参数区分QAM族
Param_R = np.zeros(n_examples)
for i in range(n_examples):
    Param_R[i] = np.max(M[i][0])/np.min(M[i][0])
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


#两个网络相连接
concate = tf.keras.layers.Concatenate()([extra_Param, output_2])
print(concate.shape)
concate = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense")(concate)
concate = tf.keras.layers.Dropout(dr)(concate)
print(concate.shape)
concate = tf.keras.layers.Dense(len(classes), kernel_initializer='he_normal', name='dense2')(concate)
concate = tf.keras.layers.Activation('softmax')(concate)
print(concate.shape)
output = tf.keras.layers.Reshape([len(classes)])(concate)
final = tf.keras.Model(inputs=[extra_Param, input_2], outputs=output)
final.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
final.summary()

nb_epoch = 100  # number of epochs to train on
batch_size = 512  # training batch size
final.load_weights("/home/baolin/PycharmProjects/AFECNN/weights.wts.h5")

# 融合特征评估
# intermediate_layer_model = tf.keras.Model(
#     inputs=final.input,
#     outputs=final.get_layer('global_average_pooling1d').output  # 替换为你实际的层名
# )
# deep_features_train = intermediate_layer_model.predict([extraData_train_tf, X_train])
# deep_features_test = intermediate_layer_model.predict([extraData_test_tf, X_test])
# combined_features_train = np.concatenate([deep_features_train, extraData_train], axis=1)
# combined_features_test = np.concatenate([deep_features_test, extraData_test], axis=1)
# scaler = StandardScaler()
# combined_features_train_scaled = scaler.fit_transform(combined_features_train)
# pca = PCA()
# X_pca = pca.fit_transform(combined_features_train_scaled)

#人工特征评估
scaler = StandardScaler()
artificial_feature_scaler = scaler.fit_transform(extraData_train)
pca = PCA()
X_pca = pca.fit_transform(artificial_feature_scaler)
class_qam = ['QAM16','QAM64','WBFM']
classes[7] = 'QAM16'
classes[8] = 'QAM64'
plt.figure(figsize=(10, 8))
for mod in class_qam:
    mod_mask = [lbl[i][0] == mod for i in train_idx]
    plt.scatter(X_pca[mod_mask, 0], X_pca[mod_mask, 1], label=mod)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Combined Deep and Handcrafted Features')
plt.legend()
plt.show()


tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(artificial_feature_scaler)
plt.figure(figsize=(12,8))
for mod in classes:
    if mod == "QAM16":
        mod_mask = [lbl[i][0] == mod for i in train_idx]
        plt.scatter(X_tsne[mod_mask, 0], X_tsne[mod_mask, 1], label="16QAM", s=2)
    elif mod == "QAM64":
        mod_mask = [lbl[i][0] == mod for i in train_idx]
        plt.scatter(X_tsne[mod_mask, 0], X_tsne[mod_mask, 1], label="64QAM", s=2)
    else:
        mod_mask = [lbl[i][0] == mod for i in train_idx]
        plt.scatter(X_tsne[mod_mask, 0], X_tsne[mod_mask, 1], label=mod, s=2)
plt.legend()
plt.title('')
plt.show()
plt.savefig(f"./t-SNE_of_artificial_features_all.png", format='png', dpi=1200,bbox_inches='tight')


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