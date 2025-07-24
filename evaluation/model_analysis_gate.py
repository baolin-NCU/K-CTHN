import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import os
import sys
from models.utils.mix_moment.mixd_moment import mixed_moment, self_moments
from models.utils.transformer.MultiHeadAttentionCustom import MultiHeadAttention, TransformerEncoder


# Artificial feature extracter - 门控版本
class FeatureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureExtractionLayer, self).__init__()
        
    def call(self, inputs):
        real = inputs[:, 0, :]  # (batch, 128)
        imag = inputs[:, 1, :]  # (batch, 128)
        comp_data = tf.complex(real, imag)  # (batch, 128)
        
        # 矩计算
        mom_20 = self_moments(comp_data, [2], axis=1)
        mom_21 = mixed_moment(comp_data, tf.math.conj(comp_data), 1, 1, axis=1)
        mom_40 = self_moments(comp_data, [4], axis=1)
        mom_41 = mixed_moment(comp_data, tf.math.conj(comp_data), 3, 1, axis=1)
        mom_42 = mixed_moment(comp_data, tf.math.conj(comp_data), 2, 2, axis=1)
        mom_60 = self_moments(comp_data, [6], axis=1)
        mom_63 = mixed_moment(comp_data, tf.math.conj(comp_data), 3, 3, axis=1)
        mom_80 = self_moments(comp_data, [8], axis=1)

        # 累积量计算
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
        Param_R = tf.reduce_max(mod_com_date, axis=1, keepdims=True) / (tf.reduce_min(mod_com_date, axis=1, keepdims=True) + 1e-8)
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
        ], axis=1)
        
        return features


def get_positional_encoding(sequence_length, d_model):
    """生成位置编码"""
    angle_rads = np.arange(sequence_length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_feature_extraction_model(input_shape):
    """创建单独的特征提取模型"""
    input_layer = tf.keras.layers.Input(shape=input_shape)
    features = FeatureExtractionLayer()(input_layer)
    return Model(inputs=input_layer, outputs=features, name="FeatureExtraction")


def create_cnn_transformer_model(input_shape):
    """创建CNN+Transformer模型"""
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    # CNN layers
    b = tf.keras.layers.Reshape(input_shape + [1])(input_layer)
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(128, (1, 3), padding='valid', activation="relu", 
                                     kernel_initializer='glorot_uniform', data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid')(b)
    
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(64, (2, 3), padding='valid', activation="relu", 
                                     kernel_initializer='glorot_uniform', data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid')(b)
    
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(32, (1, 3), padding='valid', activation="relu", 
                                     kernel_initializer='glorot_uniform', data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid')(b)
    b = tf.keras.layers.Reshape((17, 32), input_shape=(1, 17, 32))(b)
    
    # 位置编码
    pos_encoding = get_positional_encoding(sequence_length=17, d_model=32)
    b = layers.Add()([b, pos_encoding])
    
    # Transformer Encoder层 - 使用门控版本的较小配置
    transformer_block = TransformerEncoder(num_heads=2, key_dim=32, ff_dim=16, dropout=0.1)
    b = transformer_block(b, training=True)
    b = transformer_block(b, training=True)
    b = transformer_block(b, training=True)
    output = layers.GlobalAveragePooling1D()(b)
    
    return Model(inputs=input_layer, outputs=output, name="CNN_Transformer")


def create_gate_fusion_model(feature_dim, deep_dim, num_classes):
    """创建门控融合模型"""
    # 输入
    feature_input = tf.keras.layers.Input(shape=(feature_dim,), name="feature_input")
    deep_input = tf.keras.layers.Input(shape=(deep_dim,), name="deep_input")
    
    # 映射到统一维度
    output_1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_1")(feature_input)
    output_2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_2")(deep_input)
    
    # 门控融合机制
    concat = tf.keras.layers.Concatenate(axis=-1)([output_2, output_1])
    gate = tf.keras.layers.Dense(units=128, activation='sigmoid', name="gate_layer")(concat)
    fused = tf.keras.layers.Multiply(name="gated_deep")([gate, output_2])
    inv_gate = tf.keras.layers.Lambda(lambda x: 1.0 - x, name="inverse_gate")(gate)
    hand_weighted = tf.keras.layers.Multiply(name="gated_hand")([inv_gate, output_1])
    fused = tf.keras.layers.Add(name="fusion_add")([fused, hand_weighted])
    
    # 分类层
    dr = 0.5
    fused = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_3")(fused)
    fused = tf.keras.layers.Dropout(dr)(fused)
    fused = tf.keras.layers.Dense(num_classes, kernel_initializer='he_normal', name='dense_4')(fused)
    fused = tf.keras.layers.Activation('softmax')(fused)
    output = tf.keras.layers.Reshape([num_classes])(fused)
    
    return Model(inputs=[feature_input, deep_input], outputs=output, name="GateFusion")


def create_complete_gate_model(input_shape, num_classes):
    """创建完整的门控AFECNN模型"""
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    # 特征提取分支
    features = FeatureExtractionLayer()(input_layer)
    
    # CNN + Transformer 分支
    b = tf.keras.layers.Reshape(input_shape + [1])(input_layer)
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(128, (1, 3), padding='valid', activation="relu", 
                                     kernel_initializer='glorot_uniform', data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid')(b)
    
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(64, (2, 3), padding='valid', activation="relu", 
                                     kernel_initializer='glorot_uniform', data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid')(b)
    
    b = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_last")(b)
    b = tf.keras.layers.Convolution2D(32, (1, 3), padding='valid', activation="relu", 
                                     kernel_initializer='glorot_uniform', data_format="channels_last")(b)
    b = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(b)
    b = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid')(b)
    b = tf.keras.layers.Reshape((17, 32), input_shape=(1, 17, 32))(b)
    
    # 位置编码
    pos_encoding = get_positional_encoding(sequence_length=17, d_model=32)
    b = layers.Add()([b, pos_encoding])
    
    # Transformer Encoder层
    transformer_block = TransformerEncoder(num_heads=2, key_dim=32, ff_dim=16, dropout=0.1)
    b = transformer_block(b, training=True)
    b = transformer_block(b, training=True)
    b = transformer_block(b, training=True)
    output_2 = layers.GlobalAveragePooling1D()(b)
    
    # 门控融合
    output_1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_1")(features)
    output_2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_2")(output_2)
    
    concat = tf.keras.layers.Concatenate(axis=-1)([output_2, output_1])
    gate = tf.keras.layers.Dense(units=128, activation='sigmoid', name="gate_layer")(concat)
    fused = tf.keras.layers.Multiply(name="gated_deep")([gate, output_2])
    inv_gate = tf.keras.layers.Lambda(lambda x: 1.0 - x, name="inverse_gate")(gate)
    hand_weighted = tf.keras.layers.Multiply(name="gated_hand")([inv_gate, output_1])
    fused = tf.keras.layers.Add(name="fusion_add")([fused, hand_weighted])
    
    # 分类层
    dr = 0.5
    fused = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense_3")(fused)
    fused = tf.keras.layers.Dropout(dr)(fused)
    fused = tf.keras.layers.Dense(num_classes, kernel_initializer='he_normal', name='dense_4')(fused)
    fused = tf.keras.layers.Activation('softmax')(fused)
    output = tf.keras.layers.Reshape([num_classes])(fused)
    
    return Model(inputs=input_layer, outputs=output, name="AFECNN_Gate_Complete")


def get_model_parameters(model):
    """获取模型参数统计"""
    trainable_params = model.count_params()
    non_trainable_params = sum([tf.size(layer.weights[0]).numpy() for layer in model.layers 
                               if layer.weights and not layer.trainable])
    total_params = trainable_params + non_trainable_params
    
    return {
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': total_params
    }


def analyze_gate_model_complexity(input_shape=(2, 128), num_classes=11):
    """分析门控AFECNN模型复杂度"""
    print("🔍 AFECNN Gate Model Complexity Analysis")
    print("=" * 50)
    
    # 创建各个组件
    feature_model = create_feature_extraction_model(input_shape)
    cnn_transformer_model = create_cnn_transformer_model(input_shape)
    complete_model = create_complete_gate_model(input_shape, num_classes)
    
    print("\n📊 Model Component Analysis:")
    print("-" * 30)
    
    # 特征提取分支
    feature_params = get_model_parameters(feature_model)
    print(f"🔧 Feature Extraction Branch:")
    print(f"   - Parameters: {feature_params['total_params']:,}")
    print(f"   - Note: 计算复杂度高，但无可训练参数")
    
    # CNN + Transformer 分支
    cnn_params = get_model_parameters(cnn_transformer_model)
    print(f"🏗️ CNN + Transformer Branch:")
    print(f"   - Parameters: {cnn_params['total_params']:,}")
    print(f"   - Trainable: {cnn_params['trainable_params']:,}")
    
    # 完整模型
    complete_params = get_model_parameters(complete_model)
    print(f"🎯 Complete Gate Model:")
    print(f"   - Total Parameters: {complete_params['total_params']:,}")
    print(f"   - Trainable: {complete_params['trainable_params']:,}")
    print(f"   - Non-trainable: {complete_params['non_trainable_params']:,}")
    
    # 门控机制分析
    gate_layers = [layer for layer in complete_model.layers if 'gate' in layer.name.lower()]
    gate_params = sum([layer.count_params() for layer in gate_layers])
    print(f"🚪 Gate Mechanism:")
    print(f"   - Gate Parameters: {gate_params:,}")
    print(f"   - Gate Layers: {len(gate_layers)}")
    
    # 输出层分析
    print("\n📈 Layer-wise Analysis:")
    print("-" * 30)
    for i, layer in enumerate(complete_model.layers):
        if hasattr(layer, 'count_params') and layer.count_params() > 0:
            print(f"   {i:2d}. {layer.name:20s}: {layer.count_params():8,} params")
    
    # 模型架构对比
    print("\n🔄 Architecture Comparison:")
    print("-" * 30)
    print(f"   Feature Extraction: {feature_params['total_params']:8,} params (  0.0%)")
    print(f"   CNN + Transformer:  {cnn_params['total_params']:8,} params ({cnn_params['total_params']/complete_params['total_params']*100:5.1f}%)")
    print(f"   Gate + Fusion:      {complete_params['total_params'] - cnn_params['total_params']:8,} params ({(complete_params['total_params'] - cnn_params['total_params'])/complete_params['total_params']*100:5.1f}%)")
    print(f"   Total Model:        {complete_params['total_params']:8,} params (100.0%)")
    
    return {
        'feature_extraction': feature_params,
        'cnn_transformer': cnn_params,
        'complete_model': complete_params,
        'gate_params': gate_params,
        'models': {
            'feature_extraction': feature_model,
            'cnn_transformer': cnn_transformer_model,
            'complete_model': complete_model
        }
    }


if __name__ == "__main__":
    # 确保能够导入必要的模块
    try:
        results = analyze_gate_model_complexity()
        print("\n✅ Analysis completed successfully!")
        print(f"\n📋 Summary:")
        print(f"   - Total model parameters: {results['complete_model']['total_params']:,}")
        print(f"   - Gate mechanism parameters: {results['gate_params']:,}")
        print(f"   - Feature extraction: 0 trainable parameters (compute-intensive)")
        print(f"   - Main computation: CNN + Transformer branch")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("Please ensure all required dependencies are installed.")
        sys.exit(1) 