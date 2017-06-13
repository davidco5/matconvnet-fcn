
%% Change segmentations for Train set
originDirPath = 'data\Train\seg - original\';
destinationDirPath = 'data\Train\seg\';
TrainSegDir = dir('data\Train\seg\*.png');

LiverThreshold = uint8(100);
LiverSat = uint8(1);

for i = 1: numel(TrainSegDir)
    origImagePath = fullfile(originDirPath, TrainSegDir(i).name);
    segIm = imread(origImagePath);
    
    imgR = segIm(:,:,1);
    imgR( imgR > LiverThreshold) = LiverSat;
    segIm = repmat(imgR, 1,1,3);
    
    destImagePath = fullfile(destinationDirPath, TrainSegDir(i).name);
    imwrite(segIm, destImagePath); 
    
    fprintf([ num2str(i) '\n'])
end


%% Change segmentation for Validation set 

originDirPath = 'data\Val\seg - original\';
destinationDirPath = 'data\Val\seg\';
TrainSegDir = dir('data\Val\seg\*.png');

LiverThreshold = uint8(100);
LiverSat = uint8(1);

for i = 1: numel(TrainSegDir)
    origImagePath = fullfile(originDirPath, TrainSegDir(i).name);
    segIm = imread(origImagePath);
    
    imgR = segIm(:,:,1);
    imgR( imgR > LiverThreshold) = LiverSat;
    segIm = repmat(imgR, 1,1,3);
    
	destImagePath = fullfile(destinationDirPath, TrainSegDir(i).name);
    imwrite(segIm, destImagePath);    
    
     fprintf([ num2str(i) '\n'])
end

