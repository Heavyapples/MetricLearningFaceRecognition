% 1. 加载训练好的度量学习模型
load('metric_model.mat');

% 2. 准备度量学习模型的输出作为ELM的输入
numInstances = size(features, 2);
embeddedFeatures = zeros(embeddingSize, numInstances);

for i = 1:numInstances
    inputFeature = reshape(features(:, i), [1, size(features, 1), 1]);
    embeddedFeatures(:, i) = extractdata(forward(metricModel, dlarray(inputFeature, 'SSC')));
end

% 3. 划分数据集
[trainData, testData, trainLabels, testLabels] = splitData(embeddedFeatures, outputLabels);
uniqueTrainLabels = unique(trainLabels);
uniqueTestLabels = unique(testLabels);

% 4. 设置ELM参数
numHiddenNodes = 5000;
activationFunction = 'sigmoidActivation'; 
C = 1e-3;  

% 5. 训练ELM
[hiddenWeights, outputWeights] = trainELM(trainData, trainLabels, numHiddenNodes, activationFunction, C);

% 6. 测试ELM
accuracy = testELM(testData, testLabels, hiddenWeights, outputWeights, activationFunction);

% 7. 显示准确率
fprintf('ELM Classification Accuracy: %.2f%%\n', accuracy * 100);

% 8. 保存ELM的隐藏层权重和输出权重
save('elm_weights.mat', 'hiddenWeights', 'outputWeights');

