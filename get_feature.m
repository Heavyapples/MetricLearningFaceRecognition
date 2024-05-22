% 1. 加载预处理后的数据
load('preprocessed_data.mat');

% 2. 加载预训练的ResNet50网络
net = resnet50;

% 3. 准备输入和输出数据
inputData = cat(4, preprocessedData{:});
outputLabels = categorical({imgList.label});

% 4. 使用ResNet50网络进行特征提取
featureLayer = 'avg_pool';
features = activations(net, inputData, featureLayer, 'OutputAs', 'columns');

% 5. 保存提取的特征和训练的度量学习模型
save('resnet50_features.mat', 'features', 'outputLabels');