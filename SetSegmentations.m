
%% Change segmentations for Train set
dirPath = 'data\Train\seg\';
TrainSegDir = dir('data\Train\seg\*.png');

LiverThreshold = uint8(100);
LiverSat = uint8(255);

for i = 1: numel(TrainSegDir)
    imagePath = fullfile(dirPath, TrainSegDir(i).name);
    segIm = imread(imagePath);
    
    imgR = segIm(:,:,1);
    imgR( imgR > LiverThreshold) = LiverSat;
    segIm = repmat(imgR, 1,1,3);
    
    imwrite(segIm, imagePath); 
    
    fprintf([ num2str(i) '\n'])
end


%% Change segmentation for Validation set 

dirPath = 'data\Val\seg\';
TrainSegDir = dir('data\Val\seg\*.png');

LiverThreshold = uint8(100);
LiverSat = uint8(255);

for i = 1: numel(TrainSegDir)
    imagePath = fullfile(dirPath, TrainSegDir(i).name);
    segIm = imread(imagePath);
    
    imgR = segIm(:,:,1);
    imgR( imgR > LiverThreshold) = LiverSat;
    segIm(:,:,1) = imgR;
    segIm(:,:,2) = imgR;
    segIm(:,:,3) = imgR;
    
    imwrite(segIm, imagePath);    
    
     fprintf([ num2str(i) '\n'])
end

