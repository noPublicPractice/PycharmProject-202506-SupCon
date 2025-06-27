class Config(object):
    def __init__(self):
        # 模型配置
        self.input_channels = 9  # 输入通道
        self.kernel_size = 8  # 内核大小
        self.stride = 1  # 步幅
        self.final_out_channels = 128  # 最终输出通道
        self.num_classes = 6  # 类数  对应/models/model.py→第36行→self.my_logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)
        self.dropout = 0.35  # 丢弃率
        # self.features_len = 18  # 特征【源代码的】
        self.features_len = 72  # 我的添加
        # 训练配置
        self.num_epoch = 40  #【源代码的】
        # self.num_epoch = 2  # 我的添加
        # 优化器参数
        self.beta1 = 0.9  # 
        self.beta2 = 0.99  # 
        self.lr = 3e-4  # 
        # 数据参数
        self.drop_last = True  # 丢弃最后
        self.batch_size = 128  # 批量大小
        self.Context_Cont = Context_Cont_configs()  # 上下文控制
        self.TC = TC()  # 时序控制
        self.augmentation = augmentations()  # 增强
class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2  # 温度
        self.use_cosine_similarity = True  # 使用余弦相似度
class TC(object):
    def __init__(self):
        self.hidden_dim = 100  # 隐藏维度
        self.timesteps = 6  # 时间步长
class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1  # 抖动缩放比
        self.jitter_ratio = 0.8  # 抖动比
        self.max_seg = 8  # 最大分段
