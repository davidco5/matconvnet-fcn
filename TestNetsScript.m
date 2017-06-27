%%This script is meant to test the FCN-16 net

if 0
	cd('C:\Users\dcorc\OneDrive\TAU 2\Advanced Topics in Medical Image Processing 1\CNN_project\matconvnet-fcn')
    addpath(genpath(fullfile(pwd)))
    addpath(genpath('C:\Program Files\MATLAB\MatConvNet'))
    run vl_setupnn;
    dbstop if error
    load('data\fcn8_repad3\net-epoch-3.mat');
    net = dagnn.DagNN.loadobj(net) ;
    net.mode = 'test' ;
    net.removeLayer('objective') ;
    net.removeLayer('accuracy') ;
    net.vars(net.getVarIndex('bigscore')).precious = 1;
    TestNetsScript
end
load data\dataStats.mat
load data\imdb.mat
% if ~exist('dataStats', 'var')
%     if exist('data\dataStats.mat', 'file')
%         load data\dataStats.mat
%     else
%         dataStats = getDatasetStatistics(imdb);
%         save data\dataStats dataStats
%     end
% end

%% After loading the net
r = 15;
x = -r:r; y = x;
[X, Y] = meshgrid(x,y);
R = sqrt(X.^2 + Y.^2);
interval = -1*ones(size(R));
interval(R<14) = 0;
interval(R<1.4) = 1;
% imgsToRun = 1:numel(imdb.images.name);
imgsToRun = 1089;
% imgsToRun = 1092 + [1:171];
% imgsToRun = find(imdb.images.set==3);
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
    %     im_ = im_ * 0; im_(256:257,256:257) = 1;
    net.eval({'data', im_}) ;
    
    scores = net.vars(net.getVarIndex('bigscore')).value ;
    scores = squeeze(gather(scores)) ;
    
    [bestScore, best] = max(scores,[],3) ;
    probMat = exp(scores(:,:,2) - bestScore) ./ sum( exp( bsxfun(@minus, scores , bestScore) ), 3);
    predictMat = (probMat > 0.5) & dataStats.liverMask & ~dataStats.backGndSeg(imNum).seg;
    smallBlobs = bwhitmiss(uint8(predictMat), interval);
    predictMat = predictMat & ~smallBlobs;
	predictMat = imopen(uint8(predictMat), strel('disk', 4));
    if imdb.images.set(imNum) < 3
        segPath = sprintf(imdb.paths.segmentation.(dirNames{imdb.images.set(imNum)}), ['seg', imdb.images.name{imNum}]);
        seg = imread(segPath);
        seg = seg(:,:,1);
        TP = sum( predictMat(:) & seg(:) );
        FP = sum( predictMat(:) & ~seg(:) );
        FN = sum( ~predictMat(:) & seg(:) );
        sSegStats(imNum).Sens = TP / (TP + FN);
        sSegStats(imNum).PPV = TP / (TP + FP);
        sSegStats(imNum).Dice = 2*TP / (2*TP + FP + FN);
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

% im_(seg==0)=0;
% figure; imshow(im_,[]) ; colorbar
% probMat(seg==0)=0;
figure; imshow(probMat,[]) ; colorbar
figure; subplot(1,2,1); imshow(predictMat,[])
title(sprintf('Sens = %.2f, PPV = %.2f, Dice = %.2f', sSegStats(imNum).Sens, sSegStats(imNum).PPV, sSegStats(imNum).Dice), 'fontsize', 16)
subplot(1,2,2); imshow(seg(:,:,1),[])
title('Ground truth', 'fontsize', 16)
