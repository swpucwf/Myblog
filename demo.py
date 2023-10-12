import torch
import torchvision.models as models
import time
import thop
# 初始化设备
device = "cuda:0"

# 创建一个模型列表
name = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
# 确保没有残余的梯度信息
torch.manual_seed(0)
# 创建一个虚拟输入
input_data = torch.randn(1, 3, 224, 224).to(device)
# 依次加载和测试每个模型
for model_name in name:
    # 加载模型
    model = models.__getattribute__(model_name)(pretrained=True).to(device)
    # 设置模型为评估模式
    model.eval()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    # trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Total - ', pytorch_total_params)
    # print('Trainable - ', trainable_pytorch_total_params)
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    flops, params = thop.profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


    for _ in range(5):
        output = model(input_data)

    # 前向传播
    start_time = time.time()
    output = model(input_data)
    # 打印模型名称和前向传播时间
    # print(f'Inference time for {model_name}: {time.time() - start_time:.2f}s')
    inference_time = (time.time() - start_time) * 1000  # 将时间从秒转换为毫秒
    print(f'Inference time for {model_name}: {inference_time:.2f}ms')
