function [loss, gradients] = modelGradients(metricModel, anchor, positive, negative, tripletMargin, lambda)
    % 计算锚点、正样本和负样本在度量空间中的表示
    anchor_embedding = forward(metricModel, dlarray(anchor, 'SSC'));
    positive_embedding = forward(metricModel, dlarray(positive, 'SSC'));
    negative_embedding = forward(metricModel, dlarray(negative, 'SSC'));

    anchor_embedding = extractdata(anchor_embedding);
    positive_embedding = extractdata(positive_embedding);
    negative_embedding = extractdata(negative_embedding);

    % 计算triplet loss
    d_pos = sum((anchor_embedding - positive_embedding).^2, 1);
    d_neg = sum((anchor_embedding - negative_embedding).^2, 1);
    triplet_loss = max(d_pos - d_neg + tripletMargin, 0);

    % 计算 L2 正则化项
    l2_reg = 0;
    for i = 1:length(metricModel.Learnables.Value)
        l2_reg = l2_reg + lambda * sum(metricModel.Learnables.Value{i}(:).^2);
    end

    % 将 L2 正则化项添加到损失
    loss = triplet_loss + l2_reg;

    gradients = dlgradient(loss, metricModel.Learnables);
end
