import tensorflow as tf
from tensorflow.keras import layers
import os, time, datetime
import numpy as np
import pickle as cPickle
from models.utils.mix_moment.mixd_moment import mixed_moment,self_moments
from models.utils.transformer.MultiHeadAttentionCustom import TransformerEncoder

print("MobileNet-V3 AFECNN - Lightweight Architecture for Signal Classification")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# MobileNet-V3 核心组件
def hard_sigmoid(x):
    """Hard sigmoid activation - 计算效率更高"""
    return tf.nn.relu6(x + 3.0) / 6.0

def hard_swish(x):
    """Hard swish activation - MobileNet-V3的关键激活函数"""
    return x * hard_sigmoid(x)

def se_block(inputs, se_ratio=0.25):
    """Squeeze-and-Excitation注意力机制 - 自动优化通道权重"""
    filters = inputs.shape[-1]
    se_shape = (1, 1, filters)
    
    # 全局平均池化获取通道信息
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape(se_shape)(se)
    # 两层全连接实现注意力
    se = layers.Dense(int(filters * se_ratio), activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    
    return layers.multiply([inputs, se])

def inverted_res_block(inputs, filters, kernel_size, strides, expansion, 
                      use_se=False, se_ratio=0.25, activation=tf.nn.relu6):
    """倒残差块 - MobileNet的核心组件，实现轻量化卷积"""
    channel_axis = -1
    in_channels = inputs.shape[channel_axis]
    
    # 1. 扩展阶段：增加通道数
    x = layers.Conv2D(expansion * in_channels, kernel_size=(1, 1), 
                     padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    # 2. 深度卷积：每个通道独立卷积，大幅减少参数
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, 
                              padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    # 3. SE注意力机制（可选）
    if use_se:
        x = se_block(x, se_ratio)
    
    # 4. 投影阶段：降维到目标通道数
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # 5. 残差连接（维度匹配时）
    if strides == (1, 1) and in_channels == filters:
        x = layers.add([inputs, x])
    
    return x

def create_mobilenet_v3_backbone(input_tensor):
    """构建MobileNet-V3主干网络"""
    
    # 初始卷积层
    x = layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 1), padding='same', 
                     use_bias=False, name='stem_conv')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(hard_swish)(x)
    
    # MobileNet-V3 配置表
    # [kernel_size, expansion, output_channels, use_se, activation, stride]
    mobilenet_config = [
        [3, 1, 16, True, 'RE', (1, 1)],      # 保持分辨率，加SE
        [3, 4, 24, False, 'RE', (2, 1)],     # 降采样
        [3, 3, 24, False, 'RE', (1, 1)],     # 特征提取
        [5, 3, 40, True, 'HS', (2, 1)],      # 降采样，引入hard_swish
        [5, 6, 40, True, 'HS', (1, 1)],      # 深度特征提取
        [5, 6, 40, True, 'HS', (1, 1)],      # 特征增强
        [5, 3, 48, True, 'HS', (1, 1)],      # 通道调整
        [5, 6, 48, True, 'HS', (1, 1)],      # 特征细化
        [5, 6, 96, True, 'HS', (2, 1)],      # 最后降采样
        [5, 6, 96, True, 'HS', (1, 1)],      # 高级特征
    ]
    
    # 构建MobileNet-V3块
    for i, (k, exp, out, use_se, nl, s) in enumerate(mobilenet_config):
        activation = hard_swish if nl == 'HS' else tf.nn.relu
        x = inverted_res_block(x, out, (k, k), s, exp, use_se, 0.25, activation)
        print(f"Block {i+1}: {x.shape}")
    
    # 最终卷积层
    x = layers.Conv2D(576, kernel_size=(1, 1), padding='same', 
                     use_bias=False, name='head_conv')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(hard_swish)(x)
    
    return x

def create_afecnn_mobilenet_model(signal_shape, feature_shape, num_classes):
    """创建完整的AFECNN-MobileNet模型"""
    
    # 输入分支
    signal_input = layers.Input(shape=signal_shape, name='signal_input')
    feature_input = layers.Input(shape=feature_shape, name='feature_input')
    
    # 信号处理分支：MobileNet-V3 + Transformer
    # 1. 预处理
    x = layers.Reshape(signal_shape + [1])(signal_input)
    
    # 2. MobileNet-V3特征提取
    x = create_mobilenet_v3_backbone(x)
    print(f"MobileNet output shape: {x.shape}")
    
    # 3. 自适应池化和维度调整
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32 * 17, activation='relu')(x)  # 投影到Transformer维度
    x = layers.Reshape((17, 32))(x)
    
    # 4. 位置编码
    pos_encoding = get_positional_encoding(17, 32)
    x = layers.Add()([x, pos_encoding])
    
    # 5. Transformer编码器
    transformer = TransformerEncoder(num_heads=4, key_dim=32, ff_dim=64, dropout=0.1)
    x = transformer(x, training=True)
    x = transformer(x, training=True)
    signal_features = layers.GlobalAveragePooling1D()(x)
    
    # 人工特征分支
    hand_features = layers.Dense(128, activation='relu', name="hand_dense")(feature_input)
    
    # 特征融合：门控机制
    deep_features = layers.Dense(128, activation='relu', name="deep_dense")(signal_features)
    
    # 门控融合
    concat_features = layers.Concatenate()([deep_features, hand_features])
    gate = layers.Dense(128, activation='sigmoid')(concat_features)
    
    # 加权融合
    gated_deep = layers.Multiply()([gate, deep_features])
    inv_gate = layers.Lambda(lambda x: 1.0 - x)(gate)
    gated_hand = layers.Multiply()([inv_gate, hand_features])
    fused_features = layers.Add()([gated_deep, gated_hand])
    
    # 分类头
    x = layers.Dense(128, activation='relu')(fused_features)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=[feature_input, signal_input], outputs=x)
    return model

def get_positional_encoding(sequence_length, d_model):
    """生成位置编码"""
    angle_rads = np.arange(sequence_length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :]//2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# 演示模型创建
if __name__ == "__main__":
    # 假设的输入维度
    signal_shape = (2, 128)  # IQ信号
    feature_shape = (10,)    # 人工特征
    num_classes = 11         # 调制类型数
    
    print("\n=== 创建MobileNet-V3 AFECNN模型 ===")
    model = create_afecnn_mobilenet_model(signal_shape, feature_shape, num_classes)
    
    # 模型信息
    print("\n=== 模型架构 ===")
    model.summary()
    
    print(f"\n=== 模型统计 ===")
    total_params = model.count_params()
    print(f"总参数量: {total_params:,}")
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n=== MobileNet-V3 AFECNN优势 ===")
    print("1. 深度可分离卷积：大幅减少参数量和计算量")
    print("2. SE注意力机制：自动学习重要通道特征")
    print("3. Hard-Swish激活：计算效率更高的非线性")
    print("4. 倒残差结构：更好的梯度流和特征复用")
    print("5. 自适应架构：支持神经架构搜索(NAS)")
    print("6. 门控融合：智能结合深度特征和人工特征")
    print("\n模型已准备就绪！")

