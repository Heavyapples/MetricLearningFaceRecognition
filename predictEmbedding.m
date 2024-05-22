function output = predictEmbedding(model, input)
    inputLayer = model.Layers(1).Name;
    outputLayer = model.Layers(end).Name;
    tempModel = removeLayers(model, outputLayer);
    output = predict(tempModel, input);
end