function [trainData, testData, trainLabels, testLabels] = splitData(data, labels)
    % 设置数据分割参数
    trainRatio = 0.8;
    numInstances = size(data, 2);
    numTrainInstances = round(trainRatio * numInstances);

    % 随机打乱数据
    indices = randperm(numInstances);
    trainIndices = indices(1:numTrainInstances);
    testIndices = indices(numTrainInstances + 1:end);

    % 分割数据
    trainData = data(:, trainIndices);
    testData = data(:, testIndices);
    trainLabels = labels(trainIndices);
    testLabels = labels(testIndices);
end
