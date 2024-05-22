function learnables = updateLearnables(learnables, gradients, learningRate)
    for i = 1:height(learnables)
        learnables.Value{i} = learnables.Value{i} - learningRate * gradients.Value{i};
    end
end
