%%This script is meant to test the FCN-16 net

if 0
	cd('C:\Users\dcorc\OneDrive\TAU 2\Advanced Topics in Medical Image Processing 1\CNN_project\matconvnet-fcn')
    addpath(genpath(fullfile(pwd)))
    addpath(genpath('C:\Program Files\MATLAB\MatConvNet'))
    run vl_setupnn;
    dbstop if error
    load('data\fcn8_1\net-epoch-19.mat');
    net = dagnn.DagNN.loadobj(net) ;
end
load data\imdb.mat
if ~exist('dataStats', 'var')
    if exist('data\dataStats.mat', 'file')
        load dataStats
    else
        dataStats = getDatasetStatistics(imdb);
        save data\dataStats dataStats
    end
end

% net.mode = 'test' ;
% net.removeLayer('objective') ;
% net.removeLayer('accuracy') ;
%% After loading the net

% im = imread('Data/Cat_Image.jpg');
% im = imread(fullfile('data\voc11\JPEGImages', ...
%             '2007_000027.jpg'));
inNum = 10;
imPath = sprintf(imdb.paths.image.train, ['ct', imdb.images.name{inNum}]);
segPath = sprintf(imdb.paths.segmentation.train, ['seg', imdb.images.name{inNum}]);
im = imread(imPath);
seg = imread(segPath);
im_ = single(im(:,:,1));
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
im_ = im_ ./ sqrt( dataStats.rgbCovariance(1) );

net.eval({'data', im_}) ;

scores = net.vars(net.getVarIndex('bigscore')).value ;
scores = squeeze(gather(scores)) ;

[bestScore, best] = max(scores,[],3) ;
probMat = exp(scores(:,:,2) - bestScore) ./ sum( exp( bsxfun(@minus, scores , bestScore) ), 3);
% figure; imshow(im_,[]) ; colorbar
figure; imshow(probMat,[]) ; colorbar
figure; imshow((probMat > 0.5) & dataStats.liverMask,[]) ; colorbar
figure; imshow(seg(:,:,1),[]) ; colorbar
