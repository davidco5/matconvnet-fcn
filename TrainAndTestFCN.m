%% Train and test FCN (DAG format)

dbstop if error

%% Load net and repalce layers parameters if needed

% Load net, set data and initialize

% run vl_setupnn;
useGpu = 0;
preTrainedNet = load('pascal-fcn16s-dag.mat');
net = dagnn.DagNN.loadobj(preTrainedNet) ;
net.mode = 'normal' ;
net.meta.cudnnOpts = {'cudnnworkspacelimit', 1.5 * 1024^3} ;


 % !!!!!    Check if INITPARAMS is required !!!!!!!! 
 
% Change 'score_fr' layer output. This layer is indexed as 36
for i = [1 2]
  p = net.getParamIndex(net.layers(36).params{i}) ;
  if i == 1
    sz = size(net.params(p).value) ;
    sz(end) = 2 ;
  else
    sz = [2 1] ;
  end
  net.params(p).value = zeros(sz, 'single') ;
end

% Change 'score2' layer. This layer is indexed as 37
for i = [1 2]
  p = net.getParamIndex(net.layers(37).params{i}) ;
  if i == 1
    sz = size(net.params(p).value) ;
    sz(end-1) = 2 ;
    sz(end) = 2 ;
  else
    sz = [2 1] ;
  end
  net.params(p).value = zeros(sz, 'single') ;
  
end

% Change 'score_pool4' layer. This layer is indexed as 38
for i = [1 2]
  p = net.getParamIndex(net.layers(38).params{i}) ;
  if i == 1
    sz = size(net.params(p).value) ;
    sz(end) = 2 ;
  else
    sz = [2 1] ;
  end
  net.params(p).value = zeros(sz, 'single') ;
end

% Change 'upsample_new' layer. This layer is indexed as 41
for i = [1 2]
  p = net.getParamIndex(net.layers(41).params{i}) ;
  if i == 1
    sz = size(net.params(p).value) ;
    sz(end) = 2 ;
    sz(end-1) = 2 ;
    filters = single(bilinear_u(32, 2, 2)) ;
    net.params(p).value = filters;
  else
    sz = [2 1] ;
    net.params(p).value = zeros(sz, 'single') ;
  end

end

% --------------  %

% Change Classes names and Descriptions 
classes = { 'Liver', 'NotLiver'};
Description = classes;

net.meta.classes.name = classes;
net.meta.classes.description = Description;

if useGpu
    net.move('gpu')
end

if exist('data\imdb.mat', 'file')
    load data\imdb.mat
else
    imdb = VocSetupLiver;
    save data\imdb imdb;
end

%%  Train Net

% dbstop if error
stats = getDatasetStatistics(imdb);

bopts.numThreads = 1 ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,2,'single') ;
bopts.rgbMean = stats.rgbMean ;
bopts.useGpu = useGpu ;

[net,stats] = cnn_train_dag(net, imdb, @(imdb,batch,setName) getBatch(imdb,batch,setName, bopts,'prefetch',nargout==0));





