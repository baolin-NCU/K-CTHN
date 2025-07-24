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

# Artificial features extraction (same as original)
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

# Calculate cumulants
C_20 = mom_20
C_21 = mom_21
C_40 = [mom_40 - 3 * mom_20**2 for mom_40, mom_20 in zip(mom_40, mom_20)]
C_41 = [mom_41 - 3 * mom_20*mom_21 for mom_41, mom_20,mom_21 in zip(mom_41, mom_20,mom_21)]
C_42 = [mom_42 - mom_20**2 - 2*mom_21**2 for mom_42, mom_20, mom_21 in zip(mom_42, mom_20, mom_21)]
C_60 = [mom_60 - 15*mom_40*mom_20 + 30*mom_20**3 for mom_60, mom_40, mom_20 in zip(mom_60, mom_40, mom_20)]
C_63 = [mom_63 - 9*C_42*C_21 - 6*C_21**3 for mom_63, C_42, C_21 in zip(mom_63, C_42, C_21)]
C_80 = [mom_80 - 28*mom_60*mom_20 - 35*mom_40**2 + 420*mom_40*mom_20**2 - 630*mom_20**4 for mom_80, mom_60,mom_20,mom_40
        in zip(mom_80, mom_60,mom_20,mom_40)]

# Feature parameters
M_1 = [(c20 / c21) for c20, c21 in zip(C_20, C_21)]
M_3 = [(C_40 / C_42) for C_40,C_42 in zip(C_40,C_42)]
M_2 = [np.abs(c42 / c21**2) for c42, c21 in zip(C_42,C_21)]
M_4 = [np.abs(c40 / c21**2) for c40, c21 in zip(C_40,C_21)]
M_5 = [np.abs(c63 / c21**3) for c63, c21 in zip(C_63,C_21)]
M_6 = [np.abs(c80 / c21**4) for c80, c21 in zip(C_80,C_21)]

extraData = [Param_R,np.real(M_1),np.imag(M_1),M_2
            ,np.real(M_3),np.imag(M_3),M_4,M_5,M_6,np.real(C_60)]
extraData_same = [np.array(item).flatten() for item in extraData]
extraData_np = np.array(extraData_same).reshape((len(extraData_same[0]),len(extraData_same)))
extraData_train = np.real(extraData_np[train_idx])
extraData_test = np.real(extraData_np[test_idx])
extraData_train_tf = tf.convert_to_tensor(extraData_train, dtype=tf.float32)
extraData_test_tf = tf.convert_to_tensor(extraData_test, dtype=tf.float32)
in_shp_2 = list(extraData_train.shape[1:])
print(extraData_train.shape,in_shp_2)
extra_Param = tf.keras.layers.Input(shape=(in_shp_2))

# MobileNet-V3 building blocks
def hard_sigmoid(x):
    """Hard sigmoid activation function used in MobileNet-V3"""
    return tf.nn.relu6(x + 3.0) / 6.0

def hard_swish(x):
    """Hard swish activation function used in MobileNet-V3"""
    return x * hard_sigmoid(x)

def se_block(inputs, se_ratio=0.25):
    """Squeeze-and-Excitation block"""
    filters = inputs.shape[-1]
    se_shape = (1, 1, filters)
    
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(int(filters * se_ratio), activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    return layers.multiply([inputs, se])

def depthwise_conv_block(inputs, pointwise_conv_filters, depthwise_conv_strides=(1, 1), 
                        activation=tf.nn.relu6, use_se=False, se_ratio=0.25):
    """Depthwise separable convolution block"""
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=depthwise_conv_strides, 
                              padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    if use_se:
        x = se_block(x, se_ratio)
    
    x = layers.Conv2D(pointwise_conv_filters, kernel_size=(1, 1), strides=(1, 1), 
                     padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    return x

def inverted_res_block(inputs, filters, kernel_size, strides, expansion, 
                      use_se=False, se_ratio=0.25, activation=tf.nn.relu6):
    """Inverted residual block (MobileNet-V2 style with SE for V3)"""
    channel_axis = -1
    in_channels = inputs.shape[channel_axis]
    
    # Expansion phase
    x = layers.Conv2D(expansion * in_channels, kernel_size=(1, 1), 
                     padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, 
                              padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    # SE block
    if use_se:
        x = se_block(x, se_ratio)
    
    # Projection phase
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Residual connection
    if strides == (1, 1) and in_channels == filters:
        x = layers.add([inputs, x])
    
    return x

def get_positional_encoding(sequence_length, d_model):
    angle_rads = np.arange(sequence_length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :]//2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# MobileNet-V3 Small backbone for signal processing
def create_mobilenet_v3_small_backbone(input_tensor):
    """Create MobileNet-V3 Small backbone adapted for signal data"""
    
    # Initial conv layer
    x = layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 1), padding='same', 
                     use_bias=False, name='Conv')(input_tensor)
    x = layers.BatchNormalization(name='Conv/BatchNorm')(x)
    x = layers.Activation(hard_swish, name='Conv/Relu')(x)
    
    # MobileNet-V3 Small blocks configuration
    # [kernel, exp, out, SE, NL, s]
    config = [
        [3, 16, 16, True, 'RE', (1, 1)],      # bneck1
        [3, 72, 24, False, 'RE', (2, 1)],     # bneck2
        [3, 88, 24, False, 'RE', (1, 1)],     # bneck3
        [5, 96, 40, True, 'HS', (2, 1)],      # bneck4
        [5, 240, 40, True, 'HS', (1, 1)],     # bneck5
        [5, 240, 40, True, 'HS', (1, 1)],     # bneck6
        [5, 120, 48, True, 'HS', (1, 1)],     # bneck7
        [5, 144, 48, True, 'HS', (1, 1)],     # bneck8
        [5, 288, 96, True, 'HS', (2, 1)],     # bneck9
        [5, 576, 96, True, 'HS', (1, 1)],     # bneck10
        [5, 576, 96, True, 'HS', (1, 1)],     # bneck11
    ]
    
    for i, (k, exp, out, use_se, nl, s) in enumerate(config):
        activation = hard_swish if nl == 'HS' else tf.nn.relu
        x = inverted_res_block(x, out, (k, k), s, exp, use_se, 0.25, activation)
    
    # Final conv layer
    x = layers.Conv2D(576, kernel_size=(1, 1), padding='same', 
                     use_bias=False, name='Conv_1')(x)
    x = layers.BatchNormalization(name='Conv_1/BatchNorm')(x)
    x = layers.Activation(hard_swish, name='Conv_1/Relu')(x)
    
    return x

# Build the model
dr = 0.5
input_2 = tf.keras.layers.Input(shape=in_shp)

# Reshape input for MobileNet processing
b = tf.keras.layers.Reshape(in_shp+[1])(input_2)

# Apply MobileNet-V3 backbone
b = create_mobilenet_v3_small_backbone(b)

# Adaptive pooling and reshape for transformer
b = layers.GlobalAveragePooling2D()(b)
b = layers.Dense(32 * 17, activation='relu')(b)  # Project to transformer input size
b = layers.Reshape((17, 32))(b)

# Add positional encoding
pos_encoding = get_positional_encoding(sequence_length=17, d_model=32)
b = layers.Add()([b, pos_encoding])

# Transformer Encoder layers
transformer_block = TransformerEncoder(num_heads=4, key_dim=32, ff_dim=64, dropout=0.1)
b = transformer_block(b, training=True)
b = transformer_block(b, training=True)
b = transformer_block(b, training=True)
output_2 = layers.GlobalAveragePooling1D(name="global_average_pooling1d")(b)

second = tf.keras.Model(input_2, output_2, name="mobilenet_v3_backbone")
second.summary()

# Optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Gate fusion mechanism (same as original)
output_1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_1")(extra_Param)
output_2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_2")(output_2)

# Weighted fusion with gating
concat = tf.keras.layers.Concatenate(axis=-1)([output_2,output_1])
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

# Training parameters
nb_epoch = 100
batch_size = 512
current_date = datetime.datetime.now().strftime("%-m-%-d-%H-%M")
save_dir = os.path.join(root_dir,"runs/AFECNN-MobileNetV3",current_date)
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

filepath3 = os.path.join(save_dir,"weights.wts.h5")

# Training
history3 = final.fit([extraData_train_tf, X_train],
                    Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=([extraData_test_tf, X_test],Y_test),
                    callbacks = [
                        tf.keras.callbacks.ModelCheckpoint(filepath3, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                    ])

print ("训练已经完成，最佳权重模型已保存到根目录！正在加载最佳权重模型....")
final.load_weights(filepath3)
print ("最佳模型加载成功！")
score = final.evaluate([extraData_test_tf, X_test], Y_test, verbose=0, batch_size=batch_size)
print("Loss: ", score[0])
print("Accuracy: ", score[1])

with open(f"{save_dir}/acc_results.txt", "w") as f:
    f.write("Loss:" + str(score[0]) + "\n" + "Accuracy:" + str(score[1]) + "\n" + str(final.summary()))

# Plot training curves
plt.figure()
plt.yticks(np.arange(0, 5.0, 0.5))
plt.ylim([0, 5.0])
plt.title('Training performance')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history3.epoch, history3.history['loss'], label='train loss+error')
plt.plot(history3.epoch, history3.history['val_loss'], label='val_error')
plt.grid(True)
plt.legend()
plt.savefig(f"{save_dir}/loss.png", format='png', dpi=1200,bbox_inches='tight')

# Model statistics
total_params = final.count_params()
print(f"Total parameters: {total_params:,}")

with open(f"{save_dir}/model_stats.log", "w") as f:
    f.write(f"MobileNet-V3 based AFECNN Model Statistics\n")
    f.write(f"Total parameters: {total_params:,}\n")
    f.write(f"Loss: {score[0]:.6f}\n")
    f.write(f"Accuracy: {score[1]:.6f}\n")

print("MobileNet-V3 AFECNN model created successfully!") 
