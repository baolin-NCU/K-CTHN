import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pickle as cPickle
import os
from models.utils.mix_moment.mixd_moment import mixed_moment, self_moments
from models.utils.transformer.MultiHeadAttentionCustom import MultiHeadAttention, TransformerEncoder

# 复制你的FeatureExtractionLayer
class FeatureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureExtractionLayer, self).__init__()
    
    def call(self, inputs):
        real = inputs[:, 0, :]  # (batch, 128)
        imag = inputs[:, 1, :]  # (batch, 128)
        comp_data = tf.complex(real, imag) # (batch, 128)
        
        # 矩计算
        mom_20 = self_moments(comp_data, [2], axis=1) # (batch, 1)
        mom_21 = mixed_moment(comp_data, tf.math.conj(comp_data), 1, 1,axis=1) # (1,1)
        mom_40 = self_moments(comp_data, [4], axis=1)
        mom_41 = mixed_moment(comp_data, tf.math.conj(comp_data), 3, 1, axis=1)
        mom_42 = mixed_moment(comp_data, tf.math.conj(comp_data), 2, 2, axis=1)
        mom_60 = self_moments(comp_data, [6], axis=1)
        mom_63 = mixed_moment(comp_data, tf.math.conj(comp_data), 3, 3, axis=1)
        mom_80 = self_moments(comp_data, [8], axis=1)

        # 计算累积量
        C_20 = mom_20
        C_21 = mom_21
        C_40 = mom_40 - 3 * tf.math.square(mom_20)
        C_41 = mom_41 - 3 * mom_20 * mom_21
        C_42 = mom_42 - tf.math.square(mom_20) - 2 * tf.math.square(mom_21)
        C_60 = mom_60 - 15 * mom_40 * mom_20 + 30 * tf.math.pow(mom_20, 3)
        C_63 = mom_63 - 9 * C_42 * C_21 - 6 * tf.math.pow(C_21, 3)
        C_80 = mom_80 - 28 * mom_60 * mom_20 - 35 * tf.math.square(mom_40) + 420 * mom_40 * tf.math.square(mom_20) - 630 * tf.math.pow(mom_20, 4)
        
        # 特征参数计算
        mod_com_date = tf.math.abs(comp_data)
        Param_R = tf.reduce_max(mod_com_date, axis=1,keepdims=True) / (tf.reduce_min(mod_com_date, axis=1,keepdims=True) + 1e-8)
        M_1 = C_20 / C_21
        M_2 = tf.math.abs(C_42 / tf.math.square(C_21))
        M_3 = C_40 / C_42
        M_4 = tf.math.abs(C_40 / tf.math.square(C_21))
        M_5 = tf.math.abs(C_63 / tf.math.pow(C_21, 3))
        M_6 = tf.math.abs(C_80 / tf.math.pow(C_21, 4))
        features = tf.concat([
            Param_R,
            tf.math.real(M_1), tf.math.imag(M_1),
            M_2,
            tf.math.real(M_3), tf.math.imag(M_3),
            M_4, M_5, M_6,
            tf.math.real(C_60), tf.math.imag(C_60)
        ], axis=1)  # 在特征维度拼接

        return features

def get_positional_encoding(sequence_length, d_model):
    angle_rads = np.arange(sequence_length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :]//2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_feature_extraction_model(input_shape):
    """创建仅包含人工特征提取的模型"""
    input_layer = tf.keras.layers.Input(shape=input_shape)
    features = FeatureExtractionLayer()(input_layer)
    return Model(inputs=input_layer, outputs=features, name="FeatureExtraction")

def create_cnn_transformer_model(input_shape):
    """创建CNN+Transformer分支模型"""
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    # CNN部分
    b = tf.keras.layers.Reshape(input_shape + [1])(input_layer)
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(128, (1, 3), padding='valid', activation="relu", 
                                      name="conv4", kernel_initializer='glorot_uniform', 
                                      data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', 
                                     data_format=None)(b)
    
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(64, (2, 3), padding='valid', activation="relu", 
                                      name="conv5", kernel_initializer='glorot_uniform', 
                                      data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', 
                                     data_format=None)(b)
    
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(32, (1, 3), padding='valid', activation="relu", 
                                      name="conv6", kernel_initializer='glorot_uniform', 
                                      data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', 
                                     data_format=None)(b)
    b = tf.keras.layers.Reshape((17, 32), input_shape=(1, 17, 32))(b)
    
    # Transformer部分
    pos_encoding = get_positional_encoding(sequence_length=17, d_model=32)
    b = layers.Add()([b, pos_encoding])
    
    transformer_block = TransformerEncoder(num_heads=4, key_dim=32, ff_dim=64, dropout=0.1)
    b = transformer_block(b, training=True)
    b = transformer_block(b, training=True)
    b = transformer_block(b, training=True)
    
    output = layers.GlobalAveragePooling1D()(b)
    
    return Model(inputs=input_layer, outputs=output, name="CNN_Transformer")

def create_complete_model(input_shape, num_classes):
    """创建完整的模型"""
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    # 人工特征提取分支
    features = FeatureExtractionLayer()(input_layer)
    
    # CNN+Transformer分支
    b = tf.keras.layers.Reshape(input_shape + [1])(input_layer)
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(128, (1, 3), padding='valid', activation="relu", 
                                      name="conv4", kernel_initializer='glorot_uniform', 
                                      data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', 
                                     data_format=None)(b)
    
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(64, (2, 3), padding='valid', activation="relu", 
                                      name="conv5", kernel_initializer='glorot_uniform', 
                                      data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', 
                                     data_format=None)(b)
    
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(32, (1, 3), padding='valid', activation="relu", 
                                      name="conv6", kernel_initializer='glorot_uniform', 
                                      data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', 
                                     data_format=None)(b)
    b = tf.keras.layers.Reshape((17, 32), input_shape=(1, 17, 32))(b)
    
    # 添加位置编码
    pos_encoding = get_positional_encoding(sequence_length=17, d_model=32)
    b = layers.Add()([b, pos_encoding])
    
    # Transformer Encoder层
    transformer_block = TransformerEncoder(num_heads=4, key_dim=32, ff_dim=64, dropout=0.1)
    b = transformer_block(b, training=True)
    b = transformer_block(b, training=True)
    b = transformer_block(b, training=True)
    output = layers.GlobalAveragePooling1D()(b)
    
    # 两个网络相连接
    dr = 0.5
    concate = tf.keras.layers.Concatenate()([features, output])
    concate = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal', 
                                   name="dense")(concate)
    concate = tf.keras.layers.Dropout(dr)(concate)
    concate = tf.keras.layers.Dense(num_classes, kernel_initializer='he_normal', 
                                   name='dense2')(concate)
    concate = tf.keras.layers.Activation('softmax')(concate)
    output = tf.keras.layers.Reshape([num_classes])(concate)
    
    return Model(inputs=input_layer, outputs=output, name="AFECNN_Complete")

def get_model_complexity(model, input_shape):
    """计算模型复杂度"""
    # 参数量
    trainable_params = model.count_params()
    
    # 使用tf.profiler计算FLOPs
    @tf.function
    def forward_pass():
        x = tf.random.normal((1,) + input_shape)
        return model(x, training=False)
    
    # 获取具体函数
    concrete_func = forward_pass.get_concrete_function()
    
    # 计算FLOPs
    try:
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            concrete_func.graph, run_meta=run_meta, cmd='op', options=opts)
        total_flops = flops.total_float_ops
    except:
        # 如果上面的方法失败，使用近似计算
        total_flops = estimate_flops(model, input_shape)
    
    return trainable_params, total_flops

def estimate_flops(model, input_shape):
    """估算FLOPs (如果tf.profiler不可用)"""
    total_flops = 0
    
    # 遍历模型的每一层
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # 卷积层FLOPs = 输出特征图大小 × 卷积核参数量
            output_size = np.prod(layer.output_shape[1:])
            kernel_flops = np.prod(layer.kernel_size) * layer.input_shape[-1] * layer.filters
            total_flops += output_size * kernel_flops
            
        elif isinstance(layer, tf.keras.layers.Dense):
            # 全连接层FLOPs = 输入维度 × 输出维度
            if hasattr(layer, 'units'):
                input_dim = layer.input_shape[-1] if layer.input_shape[-1] else 1
                total_flops += input_dim * layer.units
    
    return total_flops

def calculate_feature_extraction_flops(input_shape):
    """计算人工特征提取的FLOPs"""
    batch_size = 1
    sequence_length = input_shape[1]  # 128
    
    # 复数操作
    complex_ops = sequence_length  # 复数构造
    
    # 各种矩计算的FLOPs估算
    moment_flops = 0
    
    # 2阶矩：每个样本需要计算x^2的和
    moment_flops += sequence_length * 2  # self_moments [2]
    
    # 混合矩计算
    moment_flops += sequence_length * 4  # mixed_moment (1,1)
    moment_flops += sequence_length * 4  # self_moments [4]
    moment_flops += sequence_length * 8  # mixed_moment (3,1)
    moment_flops += sequence_length * 8  # mixed_moment (2,2)
    moment_flops += sequence_length * 6  # self_moments [6]
    moment_flops += sequence_length * 12  # mixed_moment (3,3)
    moment_flops += sequence_length * 8  # self_moments [8]
    
    # 累积量计算
    cumulant_flops = 50  # 估算累积量计算的运算次数
    
    # 特征参数计算
    feature_flops = 100  # 估算特征参数计算的运算次数
    
    total_flops = complex_ops + moment_flops + cumulant_flops + feature_flops
    
    return total_flops

def analyze_model_complexity():
    """分析模型复杂度"""
    input_shape = (2, 128)  # 根据你的数据形状
    num_classes = 11  # 假设11个调制方式
    
    print("="*80)
    print("AFECNN 模型复杂度分析")
    print("="*80)
    
    # 1. 人工特征提取分支
    print("\n1. 人工特征提取分支分析：")
    print("-" * 40)
    feature_model = create_feature_extraction_model(input_shape)
    
    # 人工特征提取的参数量（主要是自定义计算，无可训练参数）
    feature_params = 0
    feature_flops = calculate_feature_extraction_flops(input_shape)
    
    print(f"可训练参数量: {feature_params:,}")
    print(f"FLOPs: {feature_flops:,}")
    print(f"输出特征维度: 11 (人工特征)")
    
    # 2. CNN+Transformer分支
    print("\n2. CNN+Transformer分支分析：")
    print("-" * 40)
    cnn_transformer_model = create_cnn_transformer_model(input_shape)
    cnn_transformer_model.summary()
    
    cnn_transformer_params, cnn_transformer_flops = get_model_complexity(
        cnn_transformer_model, input_shape)
    
    print(f"可训练参数量: {cnn_transformer_params:,}")
    print(f"FLOPs: {cnn_transformer_flops:,}")
    print(f"输出特征维度: 32 (全局平均池化后)")
    
    # 3. 完整模型
    print("\n3. 完整模型分析：")
    print("-" * 40)
    complete_model = create_complete_model(input_shape, num_classes)
    complete_model.summary()
    
    complete_params, complete_flops = get_model_complexity(complete_model, input_shape)
    
    print(f"可训练参数量: {complete_params:,}")
    print(f"FLOPs: {complete_flops:,}")
    
    # 4. 汇总分析
    print("\n4. 模型复杂度汇总：")
    print("-" * 40)
    print(f"人工特征分支参数量: {feature_params:,}")
    print(f"CNN+Transformer分支参数量: {cnn_transformer_params:,}")
    print(f"融合层参数量: {complete_params - cnn_transformer_params:,}")
    print(f"总参数量: {complete_params:,}")
    print()
    print(f"人工特征分支FLOPs: {feature_flops:,}")
    print(f"CNN+Transformer分支FLOPs: {cnn_transformer_flops:,}")
    print(f"总FLOPs: {complete_flops + feature_flops:,}")
    
    # 5. 复杂度对比
    print("\n5. 各分支复杂度占比：")
    print("-" * 40)
    total_params = complete_params
    total_flops = complete_flops + feature_flops
    
    print(f"参数量占比:")
    print(f"  - 人工特征分支: {feature_params/total_params*100:.2f}%")
    print(f"  - CNN+Transformer分支: {cnn_transformer_params/total_params*100:.2f}%")
    print(f"  - 融合层: {(complete_params-cnn_transformer_params)/total_params*100:.2f}%")
    print()
    print(f"FLOPs占比:")
    print(f"  - 人工特征分支: {feature_flops/total_flops*100:.2f}%")
    print(f"  - CNN+Transformer分支: {cnn_transformer_flops/total_flops*100:.2f}%")
    print(f"  - 融合层: {(complete_flops-cnn_transformer_flops)/total_flops*100:.2f}%")
    
    return {
        'feature_params': feature_params,
        'feature_flops': feature_flops,
        'cnn_transformer_params': cnn_transformer_params,
        'cnn_transformer_flops': cnn_transformer_flops,
        'complete_params': complete_params,
        'complete_flops': complete_flops,
        'total_flops': total_flops
    }

if __name__ == "__main__":
    # 运行分析
    results = analyze_model_complexity()
    
    # 保存结果
    with open('model_complexity_analysis.pkl', 'wb') as f:
        cPickle.dump(results, f)
    
    print("\n分析结果已保存到 model_complexity_analysis.pkl") 