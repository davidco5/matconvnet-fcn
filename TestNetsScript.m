%%This script is meant to test the FCN-16 net

if 0
	cd('C:\Users\dcorc\OneDrive\TAU 2\Advanced Topics in Medical Image Processing 1\CNN_project\matconvnet-fcn')
    addpath(genpath(fullfile(pwd)))
    addpath(genpath('C:\Program Files\MATLAB\MatConvNet'))
    run vl_setupnn;
    dbstop if error
    load('data\fcn8_repad2\net-epoch-40.mat', 'net');
    net = dagnn.DagNN.loadobj(net) ;
    net.mode = 'test' ;
    net.removeLayer('objective') ;
    net.removeLayer('accuracy') ;
    net.vars(net.getVarIndex('bigscore')).precious = 1;
    TestNetsScript
end
load data\dataStats.mat
load data\imdb.mat

%% After loading the net
r = 15;
x = -r:r; y = x;
[X, Y] = meshgrid(x,y);
R = sqrt(X.^2 + Y.^2);
interval = -1*ones(size(R));
interval(R<14) = 0;
interval(R<1.4) = 1;
% imgsToRun = 1089; % 56 186 351 392 558 561 751
% load('C:\Users\dcorc\OneDrive\TAU 2\Advanced Topics in Medical Image Processing 1\CNN_project\matconvnet-fcn\data\trainStats_repad2_epoch_40.mat')
% badTrain = unique([find([sSegStats.PPV] < 0.75), find([sSegStats.Sens] < 0.75)]);
% badTrain(badTrain > 1092) = [];
% imgsToRun = badTrain;
imgsToRun = 1092 + [1:171];
% imgsToRun = find(imdb.images.set==2);
nImages = max(imgsToRun);
sSegStats = struct('TP', [], 'FP', [], 'FN', [], 'Sens', [], 'PPV', [], 'Dice', []);
sSegStats = repmat(sSegStats, 1, nImages);
dirNames = {'train', 'val', 'test'};
tic
for imNum = imgsToRun
    fprintf('computing segmentation stats for training image %d\n', imNum) ;
    imPath = sprintf(imdb.paths.image.(dirNames{imdb.images.set(imNum)}), ['ct', imdb.images.name{imNum}]);
    im = imread(imPath);
    im_ = single(im(:,:,1));
    im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
    im_ = im_ ./ sqrt( dataStats.rgbCovariance(1) );
    net.eval({'data', im_}) ;
    
    scores = net.vars(net.getVarIndex('bigscore')).value ;
    scores = squeeze(gather(scores)) ;
    
    [bestScore, best] = max(scores,[],3) ;
    probMat = exp(scores(:,:,2) - bestScore) ./ sum( exp( bsxfun(@minus, scores , bestScore) ), 3);
    predictMat = (probMat > 0.5) & dataStats.liverMask & ~dataStats.backGndSeg(imNum).seg;
    CC = bwconncomp(predictMat);
    [maxSize, largestComp] = max( cellfun(@(x) size(x,1), CC.PixelIdxList) );
%     predictMatCC = false(size(predictMat));
%     predictMatCC(CC.PixelIdxList{largestComp}) = true;
	smallBlobs = bwhitmiss(uint8(predictMat), interval);
    predictMat = predictMat & ~smallBlobs;
    if sum(predictMat(:)) > 4e3
        predictMat = imopen(uint8(predictMat), strel('disk', 4));
    else
        predictMat = imopen(uint8(predictMat), strel('disk', 2));
    end

    if imdb.images.set(imNum) < 3
        segPath = sprintf(imdb.paths.segmentation.(dirNames{imdb.images.set(imNum)}), ['seg', imdb.images.name{imNum}]);
        seg = imread(segPath);
        seg = seg(:,:,1);
        TP = sum( predictMat(:) & seg(:) );
        FP = sum( predictMat(:) & ~seg(:) );
        FN = sum( ~predictMat(:) & seg(:) );
        sSegStats(imNum).Sens = TP / (TP + FN + eps);
        sSegStats(imNum).PPV = TP / (TP + FP + eps);
        sSegStats(imNum).Dice = 2*TP / (2*TP + FP + FN + eps);
        sSegStats(imNum).TP = TP;
        sSegStats(imNum).FP = FP;
        sSegStats(imNum).FN = FN;
    else
        seg = uint8(predictMat)*255;
        segPath = sprintf(imdb.paths.segmentation.(dirNames{imdb.images.set(imNum)}), ['ct', imdb.images.name{imNum}]);
        imwrite(seg, segPath)
    end
end
toc
meanSens = mean([sSegStats(imgsToRun).Sens])
meanPPV = mean([sSegStats(imgsToRun).PPV])
meanDice = mean([sSegStats(imgsToRun).Dice])

% figure; imshow(im_,[]) ; colorbar
figure; imshow(probMat,[]) ; colorbar
figure; subplot(1,2,1); imshow(predictMat,[])
title(sprintf('Sens = %.2f, PPV = %.2f, Dice = %.2f', sSegStats(imNum).Sens, sSegStats(imNum).PPV, sSegStats(imNum).Dice), 'fontsize', 16)
subplot(1,2,2); imshow(seg(:,:,1),[])
title('Ground truth', 'fontsize', 16)
