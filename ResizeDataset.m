
%% Change segmentations for Train set
originDirPath = 'data\Train\ct\';
destinationDirPath = 'data\Train\ct - resized\';
cTrainImages = dir('data\Train\ct\*.png');

for i = 1: numel(cTrainImages)
    origImagePath = fullfile(originDirPath, cTrainImages(i).name);
    im = imread(origImagePath);
    imResized = imresize(im, 0.5, 'bicubic');
    
    destImagePath = fullfile(destinationDirPath, cTrainImages(i).name);
    imwrite(imResized, destImagePath); 
    
    fprintf([ num2str(i) '\n'])
end


%% Change segmentation for Validation set 

originDirPath = 'data\Val\ct\';
destinationDirPath = 'data\Val\ct - resized\';
cTrainImages = dir('data\Val\ct\*.png');

for i = 1: numel(cTrainImages)
    origImagePath = fullfile(originDirPath, cTrainImages(i).name);
    im = imread(origImagePath);
    imResized = imresize(im, 0.5, 'bicubic');
    
    destImagePath = fullfile(destinationDirPath, cTrainImages(i).name);
    imwrite(imResized, destImagePath); 
    
    fprintf([ num2str(i) '\n'])
end

%% Change segmentation for Test set 

originDirPath = 'data\Test\ct\';
destinationDirPath = 'data\Test\ct - resized\';
cTrainImages = dir('data\Test\ct\*.png');

for i = 1: numel(cTrainImages)
    origImagePath = fullfile(originDirPath, cTrainImages(i).name);
    im = imread(origImagePath);
    imResized = imresize(im, 0.5, 'bicubic');
    
    destImagePath = fullfile(destinationDirPath, cTrainImages(i).name);
    imwrite(imResized, destImagePath); 
    
    fprintf([ num2str(i) '\n'])
end
