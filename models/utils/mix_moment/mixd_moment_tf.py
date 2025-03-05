import tensorflow as tf

def mixed_moment(x, y, orderx, ordery, axis=None, nan_policy='propagate'):
    """
    使用 TensorFlow 计算两个随机变量的混合矩。

    参数:
      x, y: tf.Tensor
          表示两个随机变量的输入张量。
      orderx: int
          x 的幂次。
      ordery: int
          y 的幂次。
      axis: int 或 None, 可选
          指定沿哪个轴计算混合矩，默认 None 表示全部计算。
      nan_policy: {'propagate', 'omit'}, 可选
          当输入包含 NaN 时的处理策略：
            'propagate': 若存在 NaN，则结果中包含 NaN；
            'omit': 忽略 NaN 值进行计算。

    返回:
      tf.Tensor: 沿指定轴计算得到的混合矩。
    """
    if nan_policy == 'omit':
        # 构造掩码，选出非 NaN 的位置
        mask_x = ~tf.math.is_nan(x)
        mask_y = ~tf.math.is_nan(y)
        mask = tf.logical_and(mask_x, mask_y)
        # 计算乘积
        product = tf.pow(x, orderx) * tf.pow(y, ordery)
        # 对于 NaN 部分填充为 0
        product = tf.where(mask, product, tf.zeros_like(product))
        # 计算有效数据个数
        valid_count = tf.reduce_sum(tf.cast(mask, tf.float32), axis=axis)
        moment = tf.reduce_sum(product, axis=axis) / valid_count
        return moment
    else:
        return tf.reduce_mean(tf.pow(x, orderx) * tf.pow(y, ordery), axis=axis)


def self_moments(x, orders, axis=None, nan_policy='propagate'):
    """
    使用 TensorFlow 计算样本的各阶矩。

    参数:
      x: tf.Tensor
          表示随机变量的输入张量。
      orders: list 或 tuple of ints
          需要计算的矩阶列表。
      axis: int 或 None, 可选
          指定沿哪个轴计算矩，默认 None 表示全部计算。
      nan_policy: {'propagate', 'omit'}, 可选
          当输入包含 NaN 时的处理策略：
            'propagate': 若存在 NaN，则结果中包含 NaN；
            'omit': 忽略 NaN 值进行计算。

    返回:
      tf.Tensor: 按照指定阶数计算得到的矩，堆叠成一个张量。
    """
    moments = []
    if nan_policy == 'omit':
        mask = ~tf.math.is_nan(x)
        for order in orders:
            power = tf.pow(x, order)
            power = tf.where(mask, power, tf.zeros_like(power))
            valid_count = tf.reduce_sum(tf.cast(mask, tf.float32), axis=axis)
            moment = tf.reduce_sum(power, axis=axis) / valid_count
            moments.append(moment)
    else:
        for order in orders:
            moment = tf.reduce_mean(tf.pow(x, order), axis=axis)
            moments.append(moment)
    return tf.stack(moments,axis=-1)
