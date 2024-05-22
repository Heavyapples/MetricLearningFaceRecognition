function [hiddenWeights, outputWeights] = trainELM(trainData, trainLabels, numHiddenNodes, activationFunction, C)
    numFeatures = size(trainData, 1);
    uniqueLabels = unique(trainLabels);
    numClasses = numel(uniqueLabels);
    trainLabelIndices = arrayfun(@(x) find(uniqueLabels == x), trainLabels);
    trainLabelsBinary = zeros(numClasses, numel(trainLabels));
    for i = 1:numel(trainLabels)
        trainLabelsBinary(trainLabelIndices(i), i) = 1;
    end
    trainLabels = double(trainLabelsBinary);
    % 初始化随机隐藏层权重
    hiddenWeights = randn(numHiddenNodes, numFeatures);

    % 计算隐藏层输出
    hiddenLayerOutput = feval(activationFunction, hiddenWeights * trainData);

    % 计算输出层权重（使用正则化）
    outputWeights = (eye(size(hiddenLayerOutput, 1)) / C + hiddenLayerOutput * hiddenLayerOutput') \ (hiddenLayerOutput * trainLabels');
end