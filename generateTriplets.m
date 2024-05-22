function [tripletIndices] = generateTriplets(features, labels, numTriplets)
    numSamples = size(features, 2);
    uniqueLabels = unique(labels);
    numClasses = numel(uniqueLabels);

    % 计算所有样本之间的距离
    distances = pdist2(features', features');

    tripletIndices = zeros(3, numTriplets);
    for i = 1:numTriplets
        % 随机选择一个锚点样本
        anchorIdx = randi(numSamples);
        anchorLabel = labels(anchorIdx);

        % 从同一类别中选择一个正样本
        sameClassIndices = find(labels == anchorLabel);
        positiveIdx = sameClassIndices(randi(numel(sameClassIndices)));

        % 计算锚点和正样本之间的距离
        anchor_positive_distance = distances(anchorIdx, positiveIdx);

        % 从不同类别中选择一个"困难"的负样本
        diffClassIndices = find(labels ~= anchorLabel);
        diffClassDistances = distances(anchorIdx, diffClassIndices);
        % 找到一个负样本，使得其距离锚点的距离比正样本稍远一点
        hardNegativeIndices = diffClassIndices(diffClassDistances > anchor_positive_distance);
        if isempty(hardNegativeIndices)
            % 如果找不到"困难"的负样本，那么随机选择一个负样本
            negativeIdx = diffClassIndices(randi(numel(diffClassIndices)));
        else
            negativeIdx = hardNegativeIndices(randi(numel(hardNegativeIndices)));
        end

        tripletIndices(:, i) = [anchorIdx; positiveIdx; negativeIdx];
    end
end
