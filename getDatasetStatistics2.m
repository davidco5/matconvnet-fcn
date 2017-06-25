function stats = getDatasetStatistics2(imdb)

load(['data\dataStats', '', '.mat'])

stats = dataStats;
dataSet = find(imdb.images.set <= 3) ;
sBackGndSeg = repmat(struct('seg', []), numel(find(imdb.images.set > 1)), 1);
stats.backGndSeg(find(imdb.images.set == 2,1):numel(dataSet)) = sBackGndSeg;
% % Class statistics
% classCounts = zeros(2,1) ;
% info = imfinfo(fullfile(sprintf(imdb.paths.segmentation.train), ['seg', imdb.images.name{1}, '.png']));
% liverMask0 = false(info.Width);
% liverSize = zeros(1, numel(dataSet));
% for i = 1:find(imdb.images.set == 1,1,'last') 
%     fprintf('%s: computing segmentation stats for training image %d\n', mfilename, i) ;
%     lb = imread( sprintf(imdb.paths.segmentation.train, ['seg', imdb.images.name{dataSet(i)}]) );
%     ok = lb < 255 ;
%     classCounts = classCounts + accumarray(lb(ok(:))+1, 1, [2 1]) ;
%     liverMask0 = liverMask0 | lb(:,:,1);
%     liverSize(i) = sum(lb(:));
% end
% stats.classCounts = classCounts ;
% se = strel('disk',25);
% liverMask = imdilate(uint8(liverMask0), se);

% Image statistics
setName = imdb.sets.name;
for t=find(imdb.images.set == 2,1):numel(dataSet)
    fprintf('%s: computing RGB stats for %s image %d\n', mfilename, setName{imdb.images.set(t)}, t) ;
    filePath = sprintf(imdb.paths.image.(setName{imdb.images.set(t)}), ['ct', imdb.images.name{dataSet(t)}]);
    rgb = imread(filePath) ;
    rgb = single(rgb) ;
%     z = reshape(permute(rgb,[3 1 2 4]),3,[]) ;
%     n = size(z,2) ;
%     rgbm1{t} = sum(z,2)/n ;
%     rgbm2{t} = z*z'/n ;
    backGndSeg0 = regiongrowing(im2double(rgb(:,:,1)), 1, 1, 1/255);
    stats.backGndSeg(t).seg = imresize(backGndSeg0, [512,512]) > 0.5;
end
% rgbm1 = mean(cat(2,rgbm1{:}),2) ;
% rgbm2 = mean(cat(3,rgbm2{:}),3) ;
% 
% stats.rgbMean = rgbm1 ;
% stats.rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
% stats.nPixels = n;
% stats.liverMask = logical(liverMask);
% stats.liverSize = liverSize;

