%% Train and test FCN (DAG format)
[~, pcName]  = system('whoami');
if ~exist('vl_setupnn.m', 'file') && strcmp(pcName(1:end-1), 'david\dcorc')
    cd('C:\Users\dcorc\OneDrive\TAU 2\Advanced Topics in Medical Image Processing 1\CNN_project\matconvnet-fcn')
    addpath(genpath(fullfile(pwd)))
    addpath(genpath('C:\Program Files\MATLAB\MatConvNet'))
    run vl_setupnn;
    dbstop if error
end

%% Load net and repalce layers parameters if needed
net = InitNet();

if exist('data\imdb.mat', 'file')
    load data\imdb.mat
else
    imdb = VocSetupLiver;
    save data\imdb imdb;
end

%%  Train Net

% dbstop if error
if ~exist('stats', 'var')
    stats = getDatasetStatistics(imdb);
end

useGpu = 0;
net.meta.cudnnOpts = {'cudnnworkspacelimit', 1024^3} ;
bopts.numThreads = 1 ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,2,'single') ;
bopts.rgbMean = stats.rgbMean ;
bopts.useGpu = useGpu ;

tic
[net,stats] = cnn_train_dag(net, imdb, ...
    @(imdb,batch,setName) getBatch(imdb,batch,setName, bopts), ones(useGpu));
toc
