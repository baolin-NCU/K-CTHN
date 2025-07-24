import tensorflow as tf
from tensorflow.keras import layers, models
import os, time, datetime
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"] = "device=cuda,floatX=float32"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pickle as cPickle
import tensorflow_model_optimization as tfmot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import moment
from models.utils.mix_moment.mixd_moment import mixed_moment,self_moments
from models.utils.transformer.MultiHeadAttentionCustom import MultiHeadAttention, TransformerEncoder
import copy
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
# step = 400
# take = 600
# # 创建索引
# train_idx = [i for start in range(0, n_examples, step + take) for i in range(start, start + take)]
# test_idx = list(set(range(0,n_examples))-set(train_idx))
# # 验证索引总长度
# index_length = len(train_idx)

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
# M_1 = list(map(lambda x,y:np.abs(x/y), C_20,C_21))
# M_2 = list(map(lambda x,y:np.abs(x/y**2), C_42,C_21))
# M_3 = list(map(lambda x,y:np.abs(x/y), C_40,C_42))
# M_4 = list(map(lambda x,y:np.abs(x/y**2), C_40,C_21))
M_1 = [(c20 / c21) for c20, c21 in zip(C_20, C_21)]
M_3 = [(C_40 / C_42) for C_40,C_42 in zip(C_40,C_42)]
M_2 = [np.abs(c42 / c21**2) for c42, c21 in zip(C_42,C_21)]
M_4 = [np.abs(c40 / c21**2) for c40, c21 in zip(C_40,C_21)]
M_5 = [np.abs(c63 / c21**3) for c63, c21 in zip(C_63,C_21)]
M_6 = [np.abs(c80 / c21**4) for c80, c21 in zip(C_80,C_21)]
extraData = [Param_R,np.real(M_1),np.imag(M_1),M_2,
             np.real(M_3),np.imag(M_3),M_4,M_5,M_6,np.real(C_60),np.imag(C_60)]
#将子列表转换成np数组利于转换为二维
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
transformer_block = TransformerEncoder(num_heads=2, key_dim=32, ff_dim=16, dropout=0.1)
b = transformer_block(b, training=True)  # (batch_size, seq_len, key_dim)
b = transformer_block(b, training=True)  # 可堆叠多个TransformerEncoder块
b = transformer_block(b, training=True)  # 可堆叠多个TransformerEncoder块
output_2 = layers.GlobalAveragePooling1D()(b)
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


# 定义剪枝参数
batch_size = 512
nb_epoch = 50
current_date = datetime.datetime.now().strftime("%-m-%-d-%H-%M")
save_dir = os.path.join(root_dir,"runs/LM-AFECNN",current_date)
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
num_steps = np.ceil(len(X_train) / batch_size).astype(np.int32) * nb_epoch
filepath3 = os.path.join(save_dir,"weights.wts.h5")
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.1,  # 初始稀疏度从10%开始
        final_sparsity=0.7,    # 最终稀疏度70%
        begin_step=int(num_steps * 0.1),  # 前10%步数作为warmup
        end_step=int(num_steps * 0.9),    # 后10%步数保持稳定
        frequency=100,
        power=3  # 使用立方衰减加速后期剪枝
    ),
    'block_size': (1, 1),  # 结构化剪枝块大小
    'block_pooling_type': 'AVG'  # 块内平均池化
}
# 对模型进行剪枝包装
def apply_pruning_to_layers(layer):
    # 只对卷积层进行剪枝
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)) and layer.name in ["conv4", "conv5", "conv6"]:
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    return layer

# 克隆并应用剪枝
final_pruned = tf.keras.models.clone_model(
    final,
    clone_function=apply_pruning_to_layers,
)

# 重新编译模型
final_pruned.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
final_pruned.summary()

# 添加剪枝回调
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tf.keras.callbacks.ModelCheckpoint(filepath3, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
]

history3 = final_pruned.fit(
    [extraData_train_tf, X_train],
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=([extraData_test_tf, X_test], Y_test),
    callbacks=callbacks
)


# 去除剪枝标记
model_for_export = tfmot.sparsity.keras.strip_pruning(final_pruned)

model_for_export.save('pruned_model.h5')
model_for_export.summary()

# 统计每层的剪枝情况
def count_pruned_weights_by_layer(model):
    """计算每一层的权重零值比例"""
    layer_stats = []
    total_zero_weights = 0
    total_weights = 0
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            # 获取权重（不包括偏置）
            weights = layer.get_weights()[0]
            # 计算零值数量
            zero_count = np.sum(weights == 0)
            # 计算总权重数量
            total_count = weights.size
            # 计算稀疏度（零值比例）
            sparsity = zero_count / total_count if total_count > 0 else 0
            
            layer_stats.append({
                'layer_name': layer.name,
                'layer_type': layer.__class__.__name__,
                'total_weights': total_count,
                'zero_weights': zero_count,
                'sparsity': sparsity
            })
            
            total_zero_weights += zero_count
            total_weights += total_count
    
    overall_sparsity = total_zero_weights / total_weights if total_weights > 0 else 0
    
    return layer_stats, total_zero_weights, total_weights, overall_sparsity

# 打印剪枝统计结果
def print_pruning_stats(model):
    """打印模型剪枝统计信息"""
    layer_stats, total_zero_weights, total_weights, overall_sparsity = count_pruned_weights_by_layer(model)
    
    print("\n=== 模型剪枝统计 ===")
    print(f"总参数数量: {total_weights:,}")
    print(f"被剪枝(置零)的参数数量: {total_zero_weights:,}")
    print(f"总体稀疏度: {overall_sparsity:.2%}")
    print("\n各层详细剪枝情况:")
    
    # 按稀疏度从高到低排序
    layer_stats.sort(key=lambda x: x['sparsity'], reverse=True)
    
    # 打印表头
    print(f"{'层名称':<20} {'层类型':<15} {'总参数':<12} {'零值参数':<12} {'稀疏度':<10}")
    print("-" * 70)
    
    # 打印每层数据
    for stat in layer_stats:
        print(f"{stat['layer_name']:<20} {stat['layer_type']:<15} {stat['total_weights']:<12,} "
              f"{stat['zero_weights']:<12,} {stat['sparsity']:.2%}")
    
    # 保存统计信息到文件
    stats_path = os.path.join(save_dir, "pruning_stats.txt")
    with open(stats_path, "w") as f:
        f.write("=== 模型剪枝统计 ===\n")
        f.write(f"总参数数量: {total_weights:,}\n")
        f.write(f"被剪枝(置零)的参数数量: {total_zero_weights:,}\n")
        f.write(f"总体稀疏度: {overall_sparsity:.2%}\n\n")
        f.write("各层详细剪枝情况:\n")
        f.write(f"{'层名称':<20} {'层类型':<15} {'总参数':<12} {'零值参数':<12} {'稀疏度':<10}\n")
        f.write("-" * 70 + "\n")
        for stat in layer_stats:
            f.write(f"{stat['layer_name']:<20} {stat['layer_type']:<15} {stat['total_weights']:<12,} "
                  f"{stat['zero_weights']:<12,} {stat['sparsity']:.2%}\n")
    
    print(f"\n统计信息已保存到: {stats_path}")
    
    return overall_sparsity

# 打印剪枝统计信息
overall_sparsity = print_pruning_stats(model_for_export)

# 统计参数
pruned_params = model_for_export.count_params()
print(f"Total parameters after pruning: {pruned_params}")
print(f"Effective parameters after pruning: {int(pruned_params * (1 - overall_sparsity)):,} ({(1 - overall_sparsity):.2%} of original)")

from keras_flops import get_flops

flops = get_flops(
    model_for_export,
    batch_size=1  # 设置批大小为1，便于标准化 FLOPs 计算
)

print(f"Total FLOPs: {flops / 10 ** 9:.05f} GFLOPs")

# 保存 FLOPs 信息
flops_log_path = os.path.join(save_dir, "model_stats.log")
with open(flops_log_path, "w") as f:
    f.write(f"Pruned Model Parameters: {pruned_params}\n")
    f.write(f"Total FLOPs: {flops / 10 ** 9:.05f} GFLOPs\n")


model_for_export.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model_for_export.evaluate([extraData_test_tf, X_test], Y_test, verbose=0, batch_size=batch_size)
print("Loss: ", score[0])
print("Accuracy: ", score[1])



# with open('history.json', 'w') as f:
#     json.dump(history.history, f)
#     json.dump(history.epoch, f)
#
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
start_time = time.time()
test_Y_hat = final.predict([extraData_test_tf,X_test], batch_size=batch_size)
end_time = time.time()
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
plt.savefig(f"{save_dir}/confu_matrix_total.png",format='png', dpi=600,fontname='Times New Roman', bbox_inches='tight')
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
    plot_confusion_matrix(confnorm1, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    plt.savefig(f"{save_dir}/conf_Matrix_for_snr=" + str(snr)+".png", format='png', dpi=600, bbox_inches='tight')  # 设置 dpi 参数以调整保存的图像质量
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
DAE = {-20: 0.09219139999999999, -18: 0.0955434, -16: 0.1000127, -14: 0.12682859999999999, -12: 0.1704045, -10: 0.23856149999999998, -8: 0.3670543, -6: 0.529067, -4: 0.6664983999999999, -2: 0.7927566, 0: 0.8843776, 2: 0.9011374999999999, 4: 0.9167801, 6: 0.9089588000000001, 8: 0.9134281, 10: 0.9178974, 12: 0.9201321, 14: 0.9145454, 16: 0.9111933999999999, 18: 0.9190147000000001}
R_ResNet = {-20: 0.09666079999999999, -18: 0.09666079999999999, -16: 0.1033647, -14: 0.12459400000000001, -12: 0.1715218, -10: 0.2519695, -8: 0.35811570000000004, -6: 0.5011338, -4: 0.6117493, -2: 0.7167781999999999, 0: 0.7625887, 2: 0.7849353000000001, 4: 0.7938738999999999, 6: 0.8106339, 8: 0.8005779, 10: 0.8095165, 12: 0.8050472000000001, 14: 0.8151031999999999, 16: 0.7894046, 18: 0.8095165}
MCLDNN = {-20: 0.0944261, -18: 0.0944261, -16: 0.10671670000000001, -14: 0.121242, -12: 0.1614658, -10: 0.2519695, -8: 0.3871662, -6: 0.5469442999999999, -4: 0.6720851, -2: 0.8117512000000001, 0: 0.8866122, 2: 0.9078415, 4: 0.9145454, 6: 0.9123108000000001, 8: 0.9190147000000001, 10: 0.9212494, 12: 0.9190147000000001, 14: 0.9234841, 16: 0.9145454, 18: 0.9223667}
PET_CGDNN= {-20: 0.0944261, -18: 0.09330880000000001, -16: 0.1033647, -14: 0.1190073, -12: 0.1681698, -10: 0.24303080000000002, -8: 0.3860489, -6: 0.5156591, -4: 0.6318612, -2: 0.767058, 0: 0.8441537, 2: 0.8810256, 4: 0.8977854999999999, 6: 0.8955508000000001, 8: 0.8989028, 10: 0.9056068, 12: 0.9056068, 14: 0.9000202, 16: 0.9011374999999999, 18: 0.9022548}
convLSTMAE = {-20: 0.0944261, -18: 0.09107409999999999, -16: 0.0955434, -14: 0.12347670000000001, -12: 0.16593509999999997, -10: 0.23744420000000002, -8: 0.3871662, -6: 0.5480616, -4: 0.7011356000000001, -2: 0.8095165, 0: 0.8843776, 2: 0.9123108000000001, 4: 0.9201321, 6: 0.9190147000000001, 8: 0.930188, 10: 0.9223667, 12: 0.9190147000000001, 14: 0.9290707, 16: 0.9156628000000001, 18: 0.9279533999999999}
C_CNN ={-20: 0.09666079999999999, -18: 0.0977781, -16: 0.10895139999999999, -14: 0.09666079999999999, -12: 0.1368846, -10: 0.21062830000000002, -8: 0.33018250000000005, -6: 0.4877258, -4: 0.5659388, -2: 0.6910797, 0: 0.773762, 2: 0.7972258999999999, 4: 0.8184551999999999, 6: 0.8139858, 8: 0.8206897999999999, 10: 0.8218071, 12: 0.8184551999999999, 14: 0.8251591, 16: 0.8072819, 18: 0.8128685}
LSTM2 = {-20: 0.0955434, -18: 0.09666079999999999, -16: 0.1044821, -14: 0.1190073, -12: 0.1558792, -10: 0.2318575, -8: 0.3681717, -6: 0.5335363000000001, -4: 0.6687331000000001, -2: 0.7905219, 0: 0.8631483, 2: 0.9011374999999999, 4: 0.9167801, 6: 0.9111933999999999, 8: 0.9156628000000001, 10: 0.9212494, 12: 0.9201321, 14: 0.9145454, 16: 0.9134281, 18: 0.9212494}
C_ResNet = {-20: 0.0955434, -18: 0.0955434, -16: 0.1134207, -14: 0.1201247, -12: 0.1491752, -10: 0.2240362, -8: 0.30895320000000004, -6: 0.40727820000000003, -4: 0.5268323, -2: 0.6553252, 0: 0.7491808, 2: 0.7849353000000001, 4: 0.7871699, 6: 0.8117512000000001, 8: 0.8095165, 10: 0.8162205, 12: 0.8050472000000001, 14: 0.8139858, 16: 0.8016952, 18: 0.8117512000000001}
CGF_HNN = {-20: 0.0944261, -18: 0.09666079999999999, -16: 0.1055994, -14: 0.131298, -12: 0.1726391, -10: 0.26649470000000003, -8: 0.4173341, -6: 0.5961067, -4: 0.7301862, -2: 0.8430364, 0: 0.9100760999999999, 2: 0.9212494, 4: 0.9290707, 6: 0.926836, 8: 0.9290707, 10: 0.9357747000000001, 12: 0.93354, 14: 0.9324227, 16: 0.926836, 18: 0.9346572999999999}
mobileNet_based= {-20: 0.6132634457611669, -18: 0.5769404332129964, -16: 0.5879294755877035, -14: 0.5938757264193115, -12: 0.7055228610230874, -10: 0.7727473498233216, -8: 0.7726325538319055, -6: 0.8550126175728378, -4: 0.855532551184725, -2: 0.9668844334309529, 0: 0.9743473325766174, 2: 0.9716765646413887, 4: 0.9755760368663594, 6: 0.9871852517985612, 8: 0.9847727272727272, 10: 0.9864616796695732, 12: 0.9851902483481431, 14: 0.9776394651913324, 16: 0.9770975056689343, 18: 0.9541534271448026}
mobileNet_V3 = {-20: 0.097538742023701, -18: 0.09115523465703972, -16: 0.10216998191681737, -14: 0.1358962896736701, -12: 0.19284744228157538, -10: 0.2773851590106007, -8: 0.3540171335957398, -6: 0.49231475108969946, -4: 0.6038647342995169, -2: 0.6983554854697004, 0: 0.7788876276958002, 2: 0.8042485153037917, 4: 0.8230414746543778, 6: 0.8264388489208633, 8: 0.8336363636363636, 10: 0.8251491509866912, 12: 0.8213716108452951, 14: 0.8340248962655602, 16: 0.8260770975056689, 18: 0.8284157966409441}
acc_AFECNN = {-20: 0.6572470373746582, -18: 0.6067238267148014, -16: 0.6426311030741411, -14: 0.6897630755476084, -12: 0.7358533272974197, -10: 0.8005742049469965, -8: 0.8647835147024774, -6: 0.9206239963294334, -4: 0.9441936968023925, -2: 0.9815273710295112, 0: 0.9763904653802498, 2: 0.98698035632709, 4: 0.9880184331797235, 6: 0.9901079136690647, 8: 0.9886363636363636, 10: 0.9892152363469482, 12: 0.9920255183413078, 14: 0.987782388197326, 16: 0.9859410430839002, 18: 0.9832047208352247}
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

plt.plot(snrs, list(map(lambda x: acc_AFECNN[x], snrs)),
            marker='o',
            markersize=4,
            markerfacecolor='red',
            linestyle='-',
            color='red',
            label='K-CTHN')
plt.plot(snrs, list(map(lambda x: acc[x], snrs)),
            marker='x',
            markersize=4,
            markerfacecolor='hotpink',
            linestyle='-',
            color='hotpink',
            label='LW-K-CTHN')
plt.plot(snrs, list(map(lambda x: mobileNet_V3[x], snrs)),
            marker='o',
            markersize=4,
            markerfacecolor='orange',
            linestyle='-',
            color='orange',
            label='mobileNet_V3')
plt.plot(snrs, list(map(lambda x: mobileNet_based[x], snrs)),
            marker='o',
            markersize=4,
            markerfacecolor='dodgerblue',
            linestyle='-',
            color='dodgerblue',
            label='mobileNet_based')
plt.xlabel("Signal to Noise Ratio",fontsize=12)
plt.ylabel("Classification Accuracy",fontsize=12)
plt.title("")
plt.grid(True)
plt.legend(loc='upper left',fontsize=10)
ax_sub = inset_axes(plt.gca(), width='40%', height='40%', loc='center right')

# 在子图上绘制相同的数据或者是数据的一个子集
ax_sub.plot(snrs, list(map(lambda x: acc_AFECNN[x], snrs)),
            marker='o',
            markersize=2,
            markerfacecolor='red',
            linestyle='-',
            color='red',
            label='K-CTHN')
ax_sub.plot(snrs, list(map(lambda x: acc[x], snrs)),
            marker='x',
            markersize=2,
            markerfacecolor='hotpink',
            linestyle='-',
            color='hotpink',
            label='LW-K-CTHN')
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
ax_sub.plot(snrs, list(map(lambda x: mobileNet_V3[x], snrs)),
            marker='o',
            markersize=2,
            markerfacecolor='orange',
            linestyle='-',
            color='orange',
            label='mobileNet_V3')
ax_sub.plot(snrs, list(map(lambda x: mobileNet_based[x], snrs)),
            marker='o',
            markersize=2,
            markerfacecolor='dodgerblue',
            linestyle='-',
            color='dodgerblue',
            label='mobileNet_based')
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
plt.xlabel('Signal to Noise Ratio (SNR)',fontsize=12)
plt.ylabel('Classification Accuracy',fontsize=12)
plt.title('Accuracy by Modulation Scheme',fontsize=12)
plt.grid(True)
plt.legend(fontsize=10)
plt.show()
plt.savefig(f"{save_dir}/acc_trend_for_all_styles.png", format='png', dpi=1200)  # 设置 dpi 参数以调整保存的图像质量
