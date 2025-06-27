import os
import sys
from datetime import datetime
import numpy as np
import torch
from dataloader.dataloader import data_generator
from models.TC import TC
from models.model import base_Model
from trainer import Trainer, model_evaluate, gen_pseudo_labels
from utils import _calc_metrics, copy_Files
from utils import _logger, set_requires_grad
from config_files.CTG_FHR_Configs import Config as Configs

# 选择模式
# self_supervised   自监督         train_linear_1p  无对比的1%半监督      ft_1p                   微调（fine_tune）
# gen_pseudo_labels 生成伪标签      SupCon           监督对比学习          train_linear_SupCon_1p  有对比的1%半监督
# random_init       随机初始化      supervised       监督学习
class args:  # 模型参数
    logs_save_dir = r'experiments_logs/CTG_FHR'  # 实验结果保存目录
    ft_perc = "1p"  # 标记数据比例
    training_mode = 'train_linear_SupCon_1p'  # 选择模式
    seed = 0  # 种子值
    selected_dataset = 'CTG_FHR'  # 选择数据集: CTG_FHR, CTG_TOCO, HAR, EEG, Epilepsy, pFD
    data_path = r'data/'  # 包含数据集的路径
    device = 'cuda:0'  # cpu 或 cuda:0
    home_path = os.getcwd()  # 主函数的文件路径
def auxiliary_work():
    # 创建文件夹
    os.makedirs(args.logs_save_dir, exist_ok=True)  # exist_ok 为 True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。
    experiment_out_dir = os.path.join(args.logs_save_dir, args.training_mode + f"_seed_{args.seed}")
    os.makedirs(experiment_out_dir, exist_ok=True)
    # 日志
    log_file_name = os.path.join(experiment_out_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug(f'Dataset: {args.selected_dataset}')
    logger.debug(f'Mode:    {args.training_mode}')
    logger.debug("Data loaded ...")
    # 返回
    return logger, experiment_out_dir
def do_work():
    # 辅助工作
    logger, experiment_out_dir = auxiliary_work()
    # 固定随机种子以实现可重复性
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    # 加载设置
    configs = Configs()
    # 加载数据集
    train_dl, valid_dl, test_dl = data_generator(os.path.join(args.data_path, args.selected_dataset), configs, args.training_mode)
    # 加载模型
    device = torch.device(args.device)  # 目标设备
    model = base_Model(configs).to(device)  # 基线模型，继承自nn.Module类
    t_c_m = TC(configs, device).to(device)  # 时序对比模型，也继承自nn.Module类 temporal
    # 条件控制
    if "train_linear" in args.training_mode or "tl" in args.training_mode:
        ### 加载此实验前的已保存模型
        if 'SupCon' not in args.training_mode:
            load_from = os.path.join(args.logs_save_dir, f"self_supervised_seed_{args.seed}", "saved_models", "ckp_last.pt")
        else:
            load_from = os.path.join(args.logs_save_dir, f"SupCon_seed_{args.seed}", "saved_models", "ckp_last.pt")
        chkpoint = torch.load(load_from, map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        # 笔记：pretrained_dict的内容："conv_block"各层（1、2、3层）的"weight"、"bias"、"running_mean"、"running_var"、"num_batches_tracked"，"my_logits"的"weight"、"bias"
        ### 根据此实验前的已保存模型，更新base_Model类的结构和参数信息
        model_dict = model.state_dict()  # 查看base_Model的结构和参数信息
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 过滤掉不必要的键
        pretrained_dict_copy = pretrained_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in ['logits']:  # 删除这些参数（例如：末尾的线性层）
                if j in i:
                    del pretrained_dict[i]
        model_dict.update(pretrained_dict)  # 更新模型的结构和参数信息
        model.load_state_dict(model_dict)  # 更新base_Model类中
        set_requires_grad(model, pretrained_dict, requires_grad=False)  # 冻结除最后一层之外的所有内容
    if "fine_tune" in args.training_mode or "ft_" in args.training_mode:
        ### 加载此实验前的已保存模型
        if 'SupCon' not in args.training_mode:
            load_from = os.path.join(args.logs_save_dir, f"self_supervised_seed_{args.seed}", "saved_models", "ckp_last.pt")
        else:
            load_from = os.path.join(args.logs_save_dir, f"SupCon_seed_{args.seed}", "saved_models", "ckp_last.pt")
        chkpoint = torch.load(load_from, map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        ### 根据此实验前的已保存模型，更新base_Model类的结构和参数信息
        pretrained_dict_copy = pretrained_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in ['logits']:
                if j in i:
                    del pretrained_dict[i]
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if args.training_mode == "gen_pseudo_labels":
        ### 根据此实验前的已保存模型，更新base_Model类的结构和参数信息
        load_from = os.path.join(args.logs_save_dir, f"ft_{args.ft_perc}_seed_{args.seed}", "saved_models", "ckp_last.pt")
        chkpoint = torch.load(load_from, map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        model.load_state_dict(pretrained_dict)
        ### 生成伪标签"pt文件"，之后直接退出程序
        gen_pseudo_labels(model, train_dl, device, os.path.join(args.data_path, args.selected_dataset))
        sys.exit(0)
    if args.training_mode == "SupCon":
        ### 根据此实验前的已保存模型，更新base_Model类的结构和参数信息
        load_from = os.path.join(args.logs_save_dir, f"ft_{args.ft_perc}_seed_{args.seed}", "saved_models", "ckp_last.pt")
        chkpoint = torch.load(load_from, map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        model.load_state_dict(pretrained_dict)
    # if args.training_mode == "random_init":
    # 训练
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)  # 基线模型优化器
    te_co_optimizer = torch.optim.Adam(t_c_m.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)  # 时序对比模型优化器 temporal
    Trainer(model, t_c_m, model_optimizer, te_co_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_out_dir, args.training_mode)
    # 输入：基线模型、时序控制模型、基线模型优化器、时序控制模型优化器、训练集、验证集、测试集、目标设备、日志、设置、实验输出目录、选择模式
    # 测试评估
    if args.training_mode != "self_supervised" and args.training_mode != "SupCon" and args.training_mode != "SupCon_pseudo":
        total_loss, total_acc, pred_labels, true_labels = model_evaluate(model, t_c_m, test_dl, device, args.training_mode)
        _calc_metrics(pred_labels, true_labels, experiment_out_dir, args.home_path)
        # 输入：基线模型、时序对比模型、测试集、目标设备、选择模式
        # 输出：损失、准确率、预测标记、真实标记
        # 笔记：_calc_metrics()的工作是把评估结果写入到文件中
if __name__ == '__main__':
    if args.training_mode == "self_supervised" or args.training_mode == "SupCon":
        copy_Files(os.path.join(args.logs_save_dir), args.selected_dataset)
    do_work()
