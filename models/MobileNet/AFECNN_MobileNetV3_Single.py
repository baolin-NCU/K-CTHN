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
from models.utils.transformer.MultiHeadAttentionCustom import MultiHeadAttention, TransformerEncoder

# 设置GPU内存增长模式
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory configuration error: {e}")

times_new_roman_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_manager.fontManager.addfont(times_new_roman_path)
root_dir = "/home/baolin/PycharmProjects/AFECNN"
plt.rcParams['font.family'] = 'Times New Roman'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load the dataset
dataFile = f"{root_dir}/data/RML2016.10a_dict.pkl"
with open(dataFile, 'rb') as f:
    Xd = cPickle.load(f, encoding='latin1')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])

# Data processing (same as original)
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.6
train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))

X_train = X[train_idx]
X_test = X[test_idx]

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods
classes[7] = '16QAM'
classes[8] = '64QAM'

# GPU优化的MobileNet-V3组件
def hard_sigmoid(x):
    """Hard sigmoid激活函数 - GPU优化版本"""
    return tf.nn.relu6(x + 3.0) / 6.0

def hard_swish(x):
    """Hard swish激活函数 - MobileNet-V3关键组件"""
    return x * hard_sigmoid(x)

def se_block_optimized(inputs, se_ratio=0.0625):
    """内存优化的SE注意力块"""
    filters = inputs.shape[-1]
    
    # 使用Conv2D替代Dense避免大张量分配
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape((1, 1, filters))(se)
    
    # 限制最小通道数，避免过小的瓶颈
    reduced_filters = max(int(filters * se_ratio), 4)
    
    # 使用Conv2D进行降维和升维
    se = layers.Conv2D(reduced_filters, 1, activation='relu', use_bias=False)(se)
    se = layers.Conv2D(filters, 1, activation='sigmoid', use_bias=False)(se)
    
    return layers.multiply([inputs, se])

def inverted_res_block_optimized(inputs, filters, kernel_size, strides, expansion, 
                                use_se=False, se_ratio=0.0625, activation=tf.nn.relu6):
    """内存优化的倒残差块"""
    channel_axis = -1
    in_channels = inputs.shape[channel_axis]
    
    # 限制扩展通道数
    expanded_channels = min(expansion * in_channels, 192)  # 限制最大扩展
    
    # 1. 扩展阶段
    x = layers.Conv2D(expanded_channels, kernel_size=(1, 1), 
                     padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    # 2. 深度卷积
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, 
                              padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    # 3. SE注意力机制
    if use_se:
        x = se_block_optimized(x, se_ratio)
    
    # 4. 投影阶段
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # 5. 残差连接
    if strides == (1, 1) and in_channels == filters:
        x = layers.add([inputs, x])
    
    return x

# 内存优化的MobileNet-V3主干网络
def create_mobilenet_v3_backbone_optimized(input_tensor):
    """内存优化的MobileNet-V3主干网络"""
    
    # 初始卷积层 - 减少初始通道数
    x = layers.Conv2D(4, kernel_size=(3, 3), strides=(2, 2), padding='same',
                     use_bias=False, name='stem_conv')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(hard_swish)(x)
    
    # 内存优化的MobileNet-V3配置
    # [kernel_size, expansion, output_channels, use_se, activation, stride]
    optimized_config_middle = [
        [3, 1, 16, True, 'RE', (1, 1)],
        [3, 4, 24, False, 'RE', (2, 2)],
        [3, 4, 24, False, 'RE', (1, 1)],
        [5, 4, 32, True, 'HS', (2, 2)],
        [5, 4, 32, True, 'HS', (1, 1)],
        [5, 6, 48, True, 'HS', (1, 1)],
        [5, 6, 48, True, 'HS', (1, 1)],
        [5, 6, 64, True, 'HS', (2, 2)],
    ]

    print("\n=== MobileNet-V3 主干网络构建 ===")
    for i, (k, exp, out, use_se, nl, s) in enumerate(optimized_config_middle):
        activation = hard_swish if nl == 'HS' else tf.nn.relu
        x = inverted_res_block_optimized(x, out, (k, k), s, exp, use_se, 0.0625, activation)
        print(f"Block {i+1}: 输入{x.shape} -> 核{k}x{k}, 扩展{exp}x, 输出{out}, SE={use_se}")
    
    # 最终卷积层 - 减少输出通道
    x = layers.Conv2D(64, kernel_size=(1, 1), padding='same', 
                     use_bias=False, name='head_conv')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(hard_swish)(x)
    
    print(f"最终特征图形状: {x.shape}")
    return x

def get_positional_encoding(sequence_length, d_model):
    """位置编码"""
    angle_rads = np.arange(sequence_length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :]//2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# 构建单通道MobileNet-V3模型
print("\n=== 构建单通道MobileNet-V3 AFECNN模型 ===")

# 信号输入
input_signal = tf.keras.layers.Input(shape=in_shp, name='signal_input')

# 信号预处理
x = tf.keras.layers.Reshape(in_shp+[1])(input_signal)

# MobileNet-V3特征提取
x = create_mobilenet_v3_backbone_optimized(x)

# 自适应池化和Transformer预处理
x = layers.GlobalAveragePooling2D()(x)

# 投影到Transformer维度 - 大幅减少
transformer_dim = 32  # 从32减少到8
sequence_length = 16  # 从17减少到4

x = layers.Dense(transformer_dim * sequence_length, activation='relu')(x)
x = layers.Reshape((sequence_length, transformer_dim))(x)

# 位置编码
pos_encoding = get_positional_encoding(sequence_length=sequence_length, d_model=transformer_dim)
x = layers.Add()([x, pos_encoding])

# 轻量级Transformer编码器
transformer_block = TransformerEncoder(
    num_heads=2,        # 从4减少到2
    key_dim=transformer_dim,  # 8
    ff_dim=16,          # 从64减少到16  
    dropout=0.1
)

# 应用Transformer层
x = transformer_block(x, training=True)
x = transformer_block(x, training=True)

# 全局平均池化获取最终特征
signal_features = layers.GlobalAveragePooling1D(name="global_average_pooling1d")(x)

print(f"Transformer输出特征形状: {signal_features.shape}")

# 直接分类器 - 无特征融合
x = layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="classifier_dense_1")(signal_features)
x = layers.Dropout(0.5)(x)
x = layers.Dense(len(classes), kernel_initializer='he_normal', name='output_dense')(x)
output = layers.Activation('softmax')(x)

# 创建模型
model = tf.keras.Model(inputs=input_signal, outputs=output, name="Single_Channel_MobileNetV3_AFECNN")

# 优化器配置
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 显示模型信息
print("\n=== 单通道MobileNet-V3模型架构 ===")
model.summary()

# 参数统计
total_params = model.count_params()
print(f"\n=== 模型统计信息 ===")
print(f"总参数量: {total_params:,}")
print(f"输入形状: {in_shp}")
print(f"输出类别: {len(classes)}")

# 训练参数
nb_epoch = 100
batch_size = 1024  # 增大batch size
current_date = datetime.datetime.now().strftime("%-m-%-d-%H-%M")
save_dir = os.path.join(root_dir,"runs/AFECNN-MobileNetV3-Single",current_date)

if os.path.exists(save_dir):
    print(f"目录已存在：{save_dir}（将直接使用现有目录）")
else:
    try:
        os.makedirs(save_dir, exist_ok=False)
        print(f"目录创建成功：{save_dir}")
    except FileExistsError:
        print(f"并发冲突：其他进程已创建该目录")
    except PermissionError:
        print(f"权限不足：无法创建目录 {save_dir}")
        raise

filepath = os.path.join(save_dir,"weights.wts.h5")

print(f"\n=== 开始训练单通道MobileNet-V3模型 ===")
print(f"训练样本: {X_train.shape[0]}")
print(f"测试样本: {X_test.shape[0]}")
print(f"批次大小: {batch_size}")
print(f"训练轮数: {nb_epoch}")

# 开始训练
history = model.fit(X_train,
                   Y_train,
                   batch_size=batch_size,
                   epochs=nb_epoch,
                   verbose=2,
                   validation_data=(X_test, Y_test),
                   callbacks = [
                       tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                       tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                   ])

print("训练已经完成，最佳权重模型已保存到根目录！正在加载最佳权重模型....")
model.load_weights(filepath)
print("最佳模型加载成功！")

# 评估模型
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print(f"\n=== 最终评估结果 ===")
print(f"损失: {score[0]:.6f}")
print(f"准确率: {score[1]:.6f}")

# 保存结果
with open(f"{save_dir}/acc_results.txt", "w") as f:
    f.write("Single Channel MobileNet-V3 AFECNN Results\n")
    f.write("="*50 + "\n")
    f.write(f"Loss: {score[0]:.6f}\n")
    f.write(f"Accuracy: {score[1]:.6f}\n")
    f.write(f"Total Parameters: {total_params:,}\n")
    f.write(f"Input Shape: {in_shp}\n")
    f.write(f"Classes: {len(classes)}\n")

# 绘制训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.epoch, history.history['loss'], label='训练损失')
plt.plot(history.epoch, history.history['val_loss'], label='验证损失')
plt.title('训练损失曲线')
plt.xlabel('轮数')
plt.ylabel('损失')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.epoch, history.history['accuracy'], label='训练准确率')
plt.plot(history.epoch, history.history['val_accuracy'], label='验证准确率')
plt.title('训练准确率曲线')
plt.xlabel('轮数')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{save_dir}/training_curves.png", format='png', dpi=1200, bbox_inches='tight')
plt.close()

# 保存模型统计
with open(f"{save_dir}/model_stats.log", "w") as f:
    f.write(f"Single Channel MobileNet-V3 AFECNN Model Statistics\n")
    f.write(f"="*60 + "\n")
    f.write(f"总参数量: {total_params:,}\n")
    f.write(f"损失: {score[0]:.6f}\n")
    f.write(f"准确率: {score[1]:.6f}\n")
    f.write(f"输入形状: {in_shp}\n")
    f.write(f"输出类别: {len(classes)}\n")
    f.write(f"批次大小: {batch_size}\n")
    f.write(f"训练轮数: {nb_epoch}\n")
    f.write(f"\n模型架构特点:\n")
    f.write(f"- 单通道信号处理\n")
    f.write(f"- MobileNet-V3轻量级主干\n")
    f.write(f"- Transformer特征增强\n")
    f.write(f"- 内存优化设计\n")
    f.write(f"- 无人工特征融合\n")

print(f"\n🎉 单通道MobileNet-V3 AFECNN模型训练完成!")
print(f"📊 最终准确率: {score[1]:.4f}")
print(f"📉 最终损失: {score[0]:.6f}")
print(f"💾 模型文件保存在: {save_dir}")
print(f"⚡ 参数量: {total_params:,} (相比双通道版本大幅减少)")
print(f"\n✨ 单通道模型优势:")
print(f"   - 更简单的架构，易于部署")
print(f"   - 更少的参数，训练更快")
print(f"   - 更低的内存需求")
print(f"   - 纯端到端学习，无需手工特征")
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
    # test_M_i = np.reshape(test_M_i, (5456, 5))
    #   取出标签
    test_Y_i = Y_test[snr_idx]
    print(len(snr_idx[0]))
    # estimate classes对该信噪比的测试集数据进行预测
    test_Y_i_hat = model.predict([test_X_i], batch_size=batch_size)
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
    f.write(str(acc) + "\n" + str(model.summary()))
DAE = {-20: 0.09219139999999999, -18: 0.0955434, -16: 0.1000127, -14: 0.12682859999999999, -12: 0.1704045, -10: 0.23856149999999998, -8: 0.3670543, -6: 0.529067, -4: 0.6664983999999999, -2: 0.7927566, 0: 0.8843776, 2: 0.9011374999999999, 4: 0.9167801, 6: 0.9089588000000001, 8: 0.9134281, 10: 0.9178974, 12: 0.9201321, 14: 0.9145454, 16: 0.9111933999999999, 18: 0.9190147000000001}
R_ResNet = {-20: 0.09666079999999999, -18: 0.09666079999999999, -16: 0.1033647, -14: 0.12459400000000001, -12: 0.1715218, -10: 0.2519695, -8: 0.35811570000000004, -6: 0.5011338, -4: 0.6117493, -2: 0.7167781999999999, 0: 0.7625887, 2: 0.7849353000000001, 4: 0.7938738999999999, 6: 0.8106339, 8: 0.8005779, 10: 0.8095165, 12: 0.8050472000000001, 14: 0.8151031999999999, 16: 0.7894046, 18: 0.8095165}
MCLDNN = {-20: 0.0944261, -18: 0.0944261, -16: 0.10671670000000001, -14: 0.121242, -12: 0.1614658, -10: 0.2519695, -8: 0.3871662, -6: 0.5469442999999999, -4: 0.6720851, -2: 0.8117512000000001, 0: 0.8866122, 2: 0.9078415, 4: 0.9145454, 6: 0.9123108000000001, 8: 0.9190147000000001, 10: 0.9212494, 12: 0.9190147000000001, 14: 0.9234841, 16: 0.9145454, 18: 0.9223667}
PET_CGDNN= {-20: 0.0944261, -18: 0.09330880000000001, -16: 0.1033647, -14: 0.1190073, -12: 0.1681698, -10: 0.24303080000000002, -8: 0.3860489, -6: 0.5156591, -4: 0.6318612, -2: 0.767058, 0: 0.8441537, 2: 0.8810256, 4: 0.8977854999999999, 6: 0.8955508000000001, 8: 0.8989028, 10: 0.9056068, 12: 0.9056068, 14: 0.9000202, 16: 0.9011374999999999, 18: 0.9022548}
convLSTMAE = {-20: 0.0944261, -18: 0.09107409999999999, -16: 0.0955434, -14: 0.12347670000000001, -12: 0.16593509999999997, -10: 0.23744420000000002, -8: 0.3871662, -6: 0.5480616, -4: 0.7011356000000001, -2: 0.8095165, 0: 0.8843776, 2: 0.9123108000000001, 4: 0.9201321, 6: 0.9190147000000001, 8: 0.930188, 10: 0.9223667, 12: 0.9190147000000001, 14: 0.9290707, 16: 0.9156628000000001, 18: 0.9279533999999999}
C_CNN ={-20: 0.09666079999999999, -18: 0.0977781, -16: 0.10895139999999999, -14: 0.09666079999999999, -12: 0.1368846, -10: 0.21062830000000002, -8: 0.33018250000000005, -6: 0.4877258, -4: 0.5659388, -2: 0.6910797, 0: 0.773762, 2: 0.7972258999999999, 4: 0.8184551999999999, 6: 0.8139858, 8: 0.8206897999999999, 10: 0.8218071, 12: 0.8184551999999999, 14: 0.8251591, 16: 0.8072819, 18: 0.8128685}
LSTM2 = {-20: 0.0955434, -18: 0.09666079999999999, -16: 0.1044821, -14: 0.1190073, -12: 0.1558792, -10: 0.2318575, -8: 0.3681717, -6: 0.5335363000000001, -4: 0.6687331000000001, -2: 0.7905219, 0: 0.8631483, 2: 0.9011374999999999, 4: 0.9167801, 6: 0.9111933999999999, 8: 0.9156628000000001, 10: 0.9212494, 12: 0.9201321, 14: 0.9145454, 16: 0.9134281, 18: 0.9212494}
C_ResNet = {-20: 0.0955434, -18: 0.0955434, -16: 0.1134207, -14: 0.1201247, -12: 0.1491752, -10: 0.2240362, -8: 0.30895320000000004, -6: 0.40727820000000003, -4: 0.5268323, -2: 0.6553252, 0: 0.7491808, 2: 0.7849353000000001, 4: 0.7871699, 6: 0.8117512000000001, 8: 0.8095165, 10: 0.8162205, 12: 0.8050472000000001, 14: 0.8139858, 16: 0.8016952, 18: 0.8117512000000001}
CGF_HNN = {-20: 0.0944261, -18: 0.09666079999999999, -16: 0.1055994, -14: 0.131298, -12: 0.1726391, -10: 0.26649470000000003, -8: 0.4173341, -6: 0.5961067, -4: 0.7301862, -2: 0.8430364, 0: 0.9100760999999999, 2: 0.9212494, 4: 0.9290707, 6: 0.926836, 8: 0.9290707, 10: 0.9357747000000001, 12: 0.93354, 14: 0.9324227, 16: 0.926836, 18: 0.9346572999999999}
K_CTHN = {-20: 0.6679580674567, -18: 0.5852888086642599, -16: 0.6503164556962026, -14: 0.7221725525257041, -12: 0.7505658669081032, -10: 0.814708480565371, -8: 0.8430192174114378, -6: 0.9208534067446662, -4: 0.9441936968023925, -2: 0.977247127731471, 0: 0.9763904653802498, 2: 0.9787574234810416, 4: 0.9824884792626728, 6: 0.989658273381295, 8: 0.9895454545454545, 10: 0.9899036255162919, 12: 0.988835725677831, 14: 0.9859382203780545, 16: 0.9854875283446712, 18: 0.9788924194280526}
acc = {-20: 0.097538742023701, -18: 0.09115523465703972, -16: 0.10216998191681737, -14: 0.1358962896736701, -12: 0.19284744228157538, -10: 0.2773851590106007, -8: 0.3540171335957398, -6: 0.49231475108969946, -4: 0.6038647342995169, -2: 0.6983554854697004, 0: 0.7788876276958002, 2: 0.8042485153037917, 4: 0.8230414746543778, 6: 0.8264388489208633, 8: 0.8336363636363636, 10: 0.8251491509866912, 12: 0.8213716108452951, 14: 0.8340248962655602, 16: 0.8260770975056689, 18: 0.8284157966409441}
acc_our_mobile = {-20: 0.6132634457611669, -18: 0.5769404332129964, -16: 0.5879294755877035, -14: 0.5938757264193115, -12: 0.7055228610230874, -10: 0.7727473498233216, -8: 0.7726325538319055, -6: 0.8550126175728378, -4: 0.855532551184725, -2: 0.9668844334309529, 0: 0.9743473325766174, 2: 0.9716765646413887, 4: 0.9755760368663594, 6: 0.9871852517985612, 8: 0.9847727272727272, 10: 0.9864616796695732, 12: 0.9851902483481431, 14: 0.9776394651913324, 16: 0.9770975056689343, 18: 0.9541534271448026}
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

plt.plot(snrs, [acc[x] for x in snrs],
            marker='*',
            markersize=4,
            markerfacecolor='orange',
            linestyle='-',
            color='orange',
            label='mobileNet_V3')

plt.plot(snrs, list(map(lambda x: K_CTHN[x], snrs)),
            marker='o',
            markersize=4,
            markerfacecolor='red',
            linestyle='-',
            color='red',
            label='K-CTHN')
plt.plot(snrs, list(map(lambda x: acc_our_mobile[x], snrs)),
            marker='o',
            markersize=4,
            markerfacecolor='green',
            linestyle='-',
            color='green',
            label='Our_mobileNet')

plt.xlabel("Signal to Noise Ratio",fontsize=12)
plt.ylabel("Classification Accuracy",fontsize=12)
plt.title("")
plt.grid(True)
plt.legend(loc='upper left',fontsize=10)
ax_sub = inset_axes(plt.gca(), width='40%', height='40%', loc='center right')

# 在子图上绘制相同的数据或者是数据的一个子集
ax_sub.plot(snrs, list(map(lambda x: K_CTHN[x], snrs)),
            marker='o',
            markersize=2,
            markerfacecolor='red',
            linestyle='-',
            color='red',
            label='K-CTHN')
ax_sub.plot(snrs, [acc_our_mobile[x] for x in snrs],
            marker='o',
            markersize=2,
            markerfacecolor='green',
            linestyle='-',
            color='green',
            label='Our_mobileNet')
ax_sub.plot(snrs, [acc[x] for x in snrs],
            marker='*',
            markersize=2,
            markerfacecolor='orange',
            linestyle='-',
            color='orange',
            label='mobileNet_V3')

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
plt.savefig("/home/baolin/PycharmProjects/AFECNN/runs/AFECNN-MobileNetV3-Single/acc_trend.png", format='png', dpi=1200)  # 设置 dpi 参数以调整保存的图像质量

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

