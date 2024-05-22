% 1. 设置数据集路径和参数
dataDir = 'E:\代码接单\基于度量学习分类器的人脸识别系统\代码\metric_learning (2)\metric_learning\ORL';
targetSize = [224, 224]; % 网络输入层的大小
numChannels = 3; % 颜色通道数
augmentFactor = 20; % 每张图像增强的数量

% 2. 获取所有子文件夹
subfolders = dir(dataDir);
subfolders = subfolders([subfolders.isdir]); % 仅保留文件夹
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % 去除'.'和'..'文件夹

% 3. 定义图像增强器
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-30,30], ...
    'RandXScale',[0.8 1.2], ...
    'RandYScale',[0.8 1.2]);

% 4. 预处理图像并进行图像增强
imgList = [];
preprocessedData = {};

for i = 1:length(subfolders)
    fprintf('Processing folder %d out of %d...\n', i, length(subfolders));

    % 获取子文件夹中的所有图像
    imgFiles = dir(fullfile(dataDir, subfolders(i).name, '*.pgm'));
    for j = 1:length(imgFiles)
        % 加载并预处理图像
        imgPath = fullfile(dataDir, subfolders(i).name, imgFiles(j).name);
        img = imread(imgPath);
        img = imresize(img, targetSize);
        
        % 确保图像具有正确的颜色通道数量
        if size(img, 3) ~= numChannels
            img = repmat(img, 1, 1, numChannels);
        end
        
        % 对图像进行增强
        for k = 1:augmentFactor
            augmentedImg = augment(imageAugmenter, img);
            % 将增强后的图像添加到数据列表中
            preprocessedData{end+1} = augmentedImg;
            imgList(end+1).name = subfolders(i).name;
            imgList(end).label = subfolders(i).name;
        end
    end
end

% 5. 保存预处理后的数据
save('preprocessed_data.mat', 'preprocessedData', 'imgList');
