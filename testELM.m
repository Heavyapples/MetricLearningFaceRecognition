function [accuracy, predictedLabelIndices] = testELM(testData, testLabels, hiddenWeights, outputWeights, activationFunction)
    % 计算隐藏层输出
    hiddenLayerOutput = feval(activationFunction, hiddenWeights * testData);

    % 计算输出层输出
    outputLayerOutput = outputWeights' * hiddenLayerOutput;

    % 预测
    [~, predictedLabelIndices] = max(outputLayerOutput, [], 1);

    % 计算准确率
    accuracy = mean(predictedLabelIndices == double(testLabels));
end
