%%This script is meant to test the FCN-16 net

% run vl_setupnn;

preTrainedNet = load('pascal-fcn16s-dag.mat');
net = dagnn.DagNN.loadobj(preTrainedNet) ;
net.mode = 'test' ;

%% After loading the net

% im = imread('Data/Cat_Image.jpg');
% im = imread(fullfile('data\voc11\JPEGImages', ...
%             '2007_000027.jpg'));
inNum = 16;
imPath = sprintf(imdb.paths.image.train, ['ct', imdb.images.name{inNum}]);
segPath = sprintf(imdb.paths.segmentation.train, ['seg', imdb.images.name{inNum}]);
im = imread(imPath);
seg = imread(segPath);
im_ = single(im);
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;

net.eval({'data', im_}) ;

scores = net.vars(net.getVarIndex('bigscore')).value ;
scores = squeeze(gather(scores)) ;

[bestScore, best] = max(scores,[],3) ;   %% NOTE!!!! This line does not work well!! should check how MAX is meant to operate
objectClass = mode(best(:)) + 1;
figure(1) ; 
clf ; 
imagesc(scores(:,:,2)) ;

title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{objectClass}, objectClass, mean(mean(bestScore(best==objectClass))))) ;