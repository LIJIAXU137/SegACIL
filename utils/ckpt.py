import torch
import utils

def save_ckpt(path, model, optimizer, best_score):
    torch.save({
        "model_state": model.state_dict(),  # 保存模型参数
        "optimizer_state": optimizer.state_dict(),  # 保存优化器参数
        "best_score": best_score,  # 保存最佳分数
        'model_architecture': model  # 保存模型架构
    }, path)


def load_ckpt(path):
    # 加载 checkpoint
    checkpoint = torch.load(path)
    
    # 直接从 checkpoint 中恢复模型
    model = checkpoint['model_architecture']  # 这是模型架构
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state'])
    
    # 返回恢复后的模型以及其他信息
    return model, checkpoint['optimizer_state'], checkpoint['best_score']
