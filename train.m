% 1. 加载特征和标签
load('resnet50_features.mat');

% 2. 设置训练参数
numEpochs = 1;
learningRate = 0.01;
numTriplets = 500;
tripletMargin = 0.2;
lambda = 0.00001;  % L2 regularization weight

% 3. 生成triplet样本
tripletIndices = generateTriplets(features, outputLabels, numTriplets);

% 4. 初始化度量学习模型
embeddingSize = 128;
embeddingLayer = fullyConnectedLayer(embeddingSize, 'Name', 'embeddingLayer');
embeddingLayer.Weights = randn(embeddingSize, size(features, 1)) * 0.01;
embeddingLayer.Bias = zeros(embeddingSize, 1);
layers = [imageInputLayer([size(features, 1), 1], 'Normalization', 'none', 'Name', 'input'); embeddingLayer];
metricModel = dlnetwork(layerGraph(layers));

% 5. 训练度量学习模型
lossValues = [];  % 初始化一个空数组来存储损失值
figure;  % 创建一个新的图像窗口

for epoch = 1:numEpochs
    fprintf('Epoch %d/%d\n', epoch, numEpochs);

    for i = 1:numTriplets
        % 获取锚点、正样本和负样本的特征
        anchor = features(:, tripletIndices(1, i));
        positive = features(:, tripletIndices(2, i));
        negative = features(:, tripletIndices(3, i));

        % 调整特征矩阵的形状以适应predict函数
        anchor = reshape(anchor, [1, size(anchor, 1), 1]);
        positive = reshape(positive, [1, size(positive, 1), 1]);
        negative = reshape(negative, [1, size(negative, 1), 1]);

        % 更新度量学习模型的参数
        [loss, gradients] = dlfeval(@modelGradients, metricModel, anchor, positive, negative, tripletMargin, lambda);
        metricModel.Learnables = updateLearnables(metricModel.Learnables, gradients, learningRate);

        % 将损失值添加到数组
        lossValues(end + 1) = loss;

        % 显示每个triplet的损失值
        if mod(i, 100) == 0
            fprintf('Triplet %d/%d: Loss = %.4f\n', i, numTriplets, loss);
        end

        % 更新损失值图像
        plot(lossValues);
        xlabel('Iteration');
        ylabel('Loss');
        title(['Epoch ', num2str(epoch), '/', num2str(numEpochs)]);
        drawnow;  % 强制更新图像
    end
end

% 6. 保存训练好的度量学习模型和嵌入大小
save('metric_model.mat', 'metricModel', 'embeddingSize');