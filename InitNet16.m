function net = InitNet16(referenceNet)
% Load net, set data and initialize

preTrainedNet = load('pascal-fcn16s-dag.mat');
net = dagnn.DagNN.loadobj(preTrainedNet) ;
net.mode = 'normal' ;
initWeightsStd = 0.005;

net.layers(1).block.size(3) = 1;
p = net.getParamIndex(net.layers(1).params{1}) ;
net.params(p).value = sum(net.params(p).value, 3);

padSizeVector = 2 * ones(1,numel(net.layers)-1);
padSizeVector([32 , 34, 36, 38, ]) = 0;     %% Specify the indices of the convolution layers that do not require padding

for i = 1:numel(net.layers)-1
    if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
        filt = net.getParamIndex(net.layers(i).params{1}) ;
        bias = net.getParamIndex(net.layers(i).params{2}) ;
%             net.params(filt).learningRate = net.params(filt).learningRate * 8 / (numel(net.layers) - i);
        net.params(bias).learningRate = 2 * net.params(filt).learningRate ;
        
        net.layers(i).block.pad = padSizeVector(i)*ones(1,4);
    end
end

net.layers(5).block.pad = [0,0,0,0];
net.layers(10).block.pad = [0,0,0,0];
net.layers(31).block.pad = [0,0,0,0];

% Change 'score_fr' layer output. This layer is indexed as 36
for i = [1 2]
    p = net.getParamIndex(net.layers(36).params{i}) ;
    if i == 1
        sz = size(net.params(p).value) ;
        sz(end) = 2 ;
    else
        sz = [2 1] ;
    end
    net.params(p).value = initWeightsStd*randn(sz, 'single') ;
end
net.layers(36).block.size = size(...
    net.params(net.getParamIndex(net.layers(36).params{1})).value) ;

% Change 'score2' layer. This layer is indexed as 37
for i = [1 2]
    p = net.getParamIndex(net.layers(37).params{i}) ;
    if i == 1
        sz = size(net.params(p).value) ;
        sz(end-1) = 2 ;
        sz(end) = 2 ;
        net.params(p).value = single(bilinear_u(4, 1, 2));
    else
        sz = [2 1] ;
        net.params(p).value = zeros(sz, 'single') ;
    end
end
net.layers(37).block.size = size(...
    net.params(net.getParamIndex(net.layers(37).params{1})).value) ;
net.layers(37).block.crop = [1 1 1 1];

% Change 'score_pool4' layer. This layer is indexed as 38
for i = [1 2]
    p = net.getParamIndex(net.layers(38).params{i}) ;
    if i == 1
        sz = size(net.params(p).value) ;
        sz(end) = 2 ;
    else
        sz = [2 1] ;
    end
    net.params(p).value = initWeightsStd*randn(sz, 'single') ;
end
net.layers(38).block.size = size(...
    net.params(net.getParamIndex(net.layers(38).params{1})).value) ;

% Change 'crop' layer. This layer is indexed as 39
net.layers(39).block.crop = [3, 3];
net.layers(39).block.inputSizes = {[38 38 2 2], [32 32 2 2]};

% Change 'upsample_new' layer. This layer is indexed as 41
for i = [1 2]
    p = net.getParamIndex(net.layers(41).params{i}) ;
    if i == 1
        sz = size(net.params(p).value) ;
        sz(end) = 2 ;
        sz(end-1) = 2 ;
        filters = single(bilinear_u(32, 2, 2)) ;
        net.params(p).value = filters;
        net.params(p).learningRate = 0.2 ;
        net.params(p).weightDecay = 0.1 ;
    else
        sz = [] ;
        net.params(p).value = [];
    end
end
net.layers(41).block.size = size(...
    net.params(net.getParamIndex(net.layers(41).params{1})).value) ;
net.layers(41).block.hasBias = false;
net.layers(41).block.crop = [8 8 8 8];
net.layers(41).block.numGroups = 2;

net.vars(net.getVarIndex('bigscore')).precious = 1 ;
net.meta.inputs.size = [512, 512, 1, 1];
net.meta.normalization.imageSize = [512, 512, 1, 1];

% -------------------------------------------------------------------------
% Losses and statistics
% -------------------------------------------------------------------------
net.removeLayer('cropx') ;

% Add loss layer
net.addLayer('objective', ...
    SegmentationLoss('loss', 'softmaxlog'), ...
    {'bigscore', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
    SegmentationAccuracy(), ...
    {'bigscore', 'label'}, 'accuracy') ;

% --------------  %

% Change Classes names and Descriptions
classes = {'NotLiver', 'Liver'};
Description = classes;

net.meta.classes.name = classes;
net.meta.classes.description = Description;
% net.meta.inputs.size(end) = 21;