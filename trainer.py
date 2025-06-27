import os
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from models.loss import NTXentLoss, SupConLoss
# from sklearn.metrics import accuracy_score

def Trainer(model, t_c_m, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    # 开始训练
    logger.debug("Training started ....")
    
    criterion = nn.CrossEntropyLoss()  # 评论家模型
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    
    for epoch in range(1, config.num_epoch + 1):
        # 训练和验证
        train_loss, train_acc = model_train(model, t_c_m, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _ = model_evaluate(model, t_c_m, valid_dl, device, training_mode)
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss)
        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')
    
    # 训练后保存模型
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {
        'model_state_dict':                model.state_dict(),
        'temporal_contr_model_state_dict': t_c_m.state_dict()
    }
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
	# 笔记：state_dict可以用来：保存模型参数、加载模型参数、查看模型的结构和参数信息
	# 笔记：state_dict存储了模型的所有可训练参数，包括：神经网络的权重（weight）、偏置项（bias），BatchNorm和LayerNorm的均值、方差
	# 笔记：只有那些参数可以训练的layer才会被保存到模型的state_dict中，如卷积层、线性层，而池化层、BN层这些本身没有参数的层是没有在这个字典中的
	# 笔记：model_state_dict.state_dict()内容如下：
	# 笔记："conv_block"各层（1、2、3层）的"weight"、"bias"、"running_mean"、"running_var"、"num_batches_tracked"
	# 笔记："my_logits"的"weight"、"bias"
	# 笔记：temporal_contr_model_state_dict.state_dict()内容如下：
	# 笔记："projection_head"的"weight"、"bias"、"running_mean"、"running_var"、"num_batches_tracked"
	# 笔记："seq_transformer"的"c_token"、"patch_to_embedding.weight"、"patch_to_embedding.bias"
	# 笔记："seq_transformer.transformer.layers"各层（0、1、2、3层）的"fn.norm.weight"、"fn.norm.bias"、"fn.fn.to_qkv.weight"、"fn.fn.to_out.0.weight"、"fn.fn.to_out.0.bias"
	# 笔记："seq_transformer.transformer.layers"各层（0、1、2、3层）的"fn.fn.net.0.weight"、"fn.fn.net.0.bias"、"fn.fn.net.3.weight"、"fn.fn.net.3.bias"
    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # 在测试集上评估
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, t_c_m, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')
    
    logger.debug("\n################## Training is Done! #########################")
def model_train(model, t_c_m, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    t_c_m.train()
    
    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):  # aug表示增强
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()
		# 笔记：在用pytorch训练模型时，通常会在遍历epochs的过程中依次用到optimizer.zero_grad(),loss.backward()和optimizer.step()三个函数
		# 笔记：总得来说，这三个函数的作用是先将梯度归零->zero_grad()，然后反向传播计算得到每个参数的梯度值->loss.backward()，最后通过梯度下降执行一步参数更新->step()
		# 笔记：zero_grad()函数会遍历模型的所有参数，通过p.grad.detach_()方法截断反向传播的梯度流，再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空。
		# 笔记：因为训练的过程通常使用mini-batch方法，所以如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前。
        if training_mode == "self_supervised" or training_mode == "SupCon":
            if training_mode == "self_supervised":  # 我的添加
                aug1 = aug1.repeat(1, 9, 1)  # 我的添加
                aug2 = aug2.repeat(1, 9, 1)  # 我的添加
                # 笔记：repeat()表示在某一维度复制
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)
            # 规范化投影特征向量
            features1 = functional.normalize(features1, dim=1)
            features2 = functional.normalize(features2, dim=1)
            # 是对称的步骤
            temp_cont_loss1, temp_cont_feat1 = t_c_m(features1, features2)  # 时序对比
            temp_cont_loss2, temp_cont_feat2 = t_c_m(features2, features1)
            if training_mode == "self_supervised":
                lambda1 = 1
                lambda2 = 0.7
                nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature, config.Context_Cont.use_cosine_similarity)
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2
            else:
                lambda1 = 0.01
                lambda2 = 0.1
                sup_contrastive_criterion = SupConLoss(device)
                supcon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + sup_contrastive_criterion(supcon_features, labels) * lambda2
        else:
            data = data.repeat(1, 9, 1)  # 我的添加：tensor在某一维度复制
            predictions, features = model(data)
            loss = criterion(predictions, labels)  # 调用交叉熵损失函数CrossEntropyLoss
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()
		# 笔记：PyTorch的反向传播(即tensor.backward())是通过autograd包来实现的，autograd包会根据tensor进行过的数学运算来自动计算其对应的梯度。
		# 笔记：具体来说，torch.tensor是autograd包的基础类，如果设置tensor的requires_grads为True，就会开始跟踪这个tensor上面的所有运算，
		# 笔记：如果做完运算后使用tensor.backward()，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。
		# 笔记：如果没有进行tensor.backward()的话，梯度值将会是None，
		# 笔记：optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是backward()方法产生的，因此loss.backward()要写在optimizer.step()之前。
		# 笔记：step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值
    # 完成循环
    total_loss = torch.tensor(total_loss).mean()
    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc
def model_evaluate(model, t_c_m, test_dl, device, training_mode):
    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], []
    else:  # 中间不计算评估，最后做
        model.eval()
        t_c_m.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = []
        total_acc = []
        outs = np.array([])
        trgs = np.array([])
        # compute loss
        with torch.no_grad():
            # 在深度学习中，模型训练和评估是两个独立但紧密相关的过程。训练时需要计算梯度来更新模型参数，但在评估阶段，梯度计算则是不必要的负担
            # torch.no_grad() 是 PyTorch 提供的一个上下文管理器，用于在其上下文块中禁用自动求导（Autograd）的功能，从而避免梯度计算和反向传播
            for data, labels, _, _ in test_dl:
                data, labels = data.float().to(device), labels.long().to(device)
                data = data.repeat(1, 9, 1)  # 我的添加
                # 笔记：to(device)表示将张量移动到指定的设备上
                predictions, features = model(data)
                
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                # 笔记：detach()表示从计算图中分离出来，返回一个新的tensor
                # 笔记：argmax()返回指定维度上张量最大值的索引
                # 笔记：eq()对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。
                loss = criterion(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                # print(pred.cpu().numpy())
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        return total_loss, total_acc, outs, trgs
def gen_pseudo_labels(model, dataloader, device, experiment_log_dir):
    model.eval()
    softmax = nn.Softmax(dim=1)
    
    # saving output data
    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []
    
    with torch.no_grad():
        for data, labels, _, _ in dataloader:
            data = data.float().to(device)
            labels = labels.view((-1)).long().to(device)
            
            # forward pass
            data = data.repeat(1, 9, 1)  # 我的添加：tensor在某一维度复制
            predictions, features = model(data)
            
            normalized_preds = softmax(predictions)
            pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())
            
            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_data.append(data)
    
    all_data = torch.cat(all_data, dim=0)
    
    data_save = dict()
    data_save["samples"] = all_data
    data_save["labels"] = torch.LongTensor(torch.from_numpy(all_pseudo_labels).long())
    file_name = f"pseudo_train_data.pt"
    torch.save(data_save, os.path.join(experiment_log_dir, file_name))
    print("Pseudo labels generated ...")
