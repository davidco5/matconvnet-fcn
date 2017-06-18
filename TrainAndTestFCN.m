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
if exist('data\imdb.mat', 'file')
    load data\imdb.mat
else
    imdb = VocSetupLiver;
    save data\imdb imdb;
end

if ~exist('dataStats', 'var')
    if exist('data\dataStats.mat', 'file')
        load dataStats
    else
        dataStats = getDatasetStatistics(imdb);
        save data\dataStats dataStats
    end
end

% referenceNet = 'data\fcn8_1\net-epoch-19.mat';
referenceNet = [];
net = InitNet8(referenceNet);

%%  Train Net

useGpu = 0;
net.meta.cudnnOpts = {'cudnnworkspacelimit', 1.5 * 1024^3} ;
bopts.numThreads = 1 ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,2,'single') ;
bopts.rgbMean = dataStats.rgbMean(1) ;
bopts.rgbStd = sqrt( dataStats.rgbCovariance(1) );
bopts.liverMask = dataStats.liverMask;
bopts.useGpu = useGpu ;
net.meta.normalization.averageImage = dataStats.rgbMean(1);

tic
[net,stats] = cnn_train_dag(net, imdb, ...
    @(imdb,batch,setName) getBatch(imdb,batch,setName, bopts), ones(useGpu));
toc
