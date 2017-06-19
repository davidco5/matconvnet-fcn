%%This script is meant to test the FCN-16 net

if 0
	cd('C:\Users\dcorc\OneDrive\TAU 2\Advanced Topics in Medical Image Processing 1\CNN_project\matconvnet-fcn')
    addpath(genpath(fullfile(pwd)))
    addpath(genpath('C:\Program Files\MATLAB\MatConvNet'))
    run vl_setupnn;
    dbstop if error
    load('data\fcn8_3\net-epoch-21.mat');
    net = dagnn.DagNN.loadobj(net) ;
    net.mode = 'test' ;
    net.removeLayer('objective') ;
    net.removeLayer('accuracy') ;
end
load data\imdb.mat
if ~exist('dataStats', 'var')
    if exist('data\dataStats.mat', 'file')
        load data\dataStats
    else
        dataStats = getDatasetStatistics(imdb);
        save data\dataStats dataStats
    end
end

%% After loading the net
% imgsToRun = 1:numel(imdb.images.name);
imgsToRun = 1093:1592;
% imgsToRun = 1;
nImages = max(imgsToRun);
sSegStats = struct('TP', [], 'FP', [], 'FN', [], 'Sens', [], 'PPV', [], 'Dice', []);
sSegStats = repmat(sSegStats, 1, nImages);
dirNames = {'train', 'val'};
for imNum = imgsToRun
    fprintf('computing segmentation stats for training image %d\n', imNum) ;
    imPath = sprintf(imdb.paths.image.(dirNames{ceil(imNum/1092)}), ['ct', imdb.images.name{imNum}]);
    segPath = sprintf(imdb.paths.segmentation.(dirNames{ceil(imNum/1092)}), ['seg', imdb.images.name{imNum}]);
    im = imread(imPath);
    seg = imread(segPath);
    seg = seg(:,:,1);
    im_ = single(im(:,:,1));
    im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
    im_ = im_ ./ sqrt( dataStats.rgbCovariance(1) );
    
    net.eval({'data', im_}) ;
    
    scores = net.vars(net.getVarIndex('bigscore')).value ;
    scores = squeeze(gather(scores)) ;
    
    [bestScore, best] = max(scores,[],3) ;
    probMat = exp(scores(:,:,2) - bestScore) ./ sum( exp( bsxfun(@minus, scores , bestScore) ), 3);
    predictMat = (probMat > 0.5) & dataStats.liverMask;
    TP = sum( predictMat(:) & seg(:) );
    FP = sum( predictMat(:) & ~seg(:) );
    FN = sum( ~predictMat(:) & seg(:) );
    sSegStats(imNum).Sens = TP / (TP + FN);
    sSegStats(imNum).PPV = TP / (TP + FP);
    sSegStats(imNum).Dice = 2*TP / (2*TP + FP + FN);
    sSegStats(imNum).TP = TP;
    sSegStats(imNum).FP = FP;
    sSegStats(imNum).FN = FN;
end

meanSens = mean([sSegStats(imgsToRun).Sens])
meanPPV = mean([sSegStats(imgsToRun).PPV])
meanDice = mean([sSegStats(imgsToRun).Dice])

% figure; imshow(im_,[]) ; colorbar
% figure; imshow(probMat,[]) ; colorbar
figure; subplot(1,2,1); imshow(predictMat,[])
title(sprintf('Sens = %.2f, PPV = %.2f, Dice = %.2f', sSegStats(imNum).Sens, sSegStats(imNum).PPV, sSegStats(imNum).Dice), 'fontsize', 16)
subplot(1,2,2); imshow(seg(:,:,1),[])
title('Ground truth', 'fontsize', 16)
