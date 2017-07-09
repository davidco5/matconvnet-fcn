function net = InitNet8(referenceNet, resizeFlag)
% Load net, set data and initialize
if ~exist('resizeFlag', 'var')
    resizeFlag = false;
end
if ~isempty(referenceNet)
    load(referenceNet, 'net')
    net = dagnn.DagNN.loadobj(net) ;
    net.mode = 'normal' ;
    net.addLayer('drop3', dagnn.DropOut('rate', 0.5), 'conv2_1x', 'conv2_1x_drop');
    net.setLayerInputs('conv2_2', {'conv2_1x_drop'});
%     net.setLayerParams('score2', {'score2_filter'});
    for i = 1:numel(net.layers)-1
    if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
        filt = net.getParamIndex(net.layers(i).params{1}) ;
        filtVal = net.params(filt).value;
        if size(filtVal,4) > 2
            net.params(filt).value(:,:,min(size(filtVal,3),60),60) = 1/size(filtVal,1) * ones(size(filtVal,1), 'single');
        end
    end
    end
    return
end

preTrainedNet = load('data\models\pascal-fcn8s-tvg-dag.mat');
net = dagnn.DagNN.loadobj(preTrainedNet) ;
net.mode = 'normal' ;
initWeightsStd = 0.005;

%% remove crop layers
net.removeLayer('crop') ;
net.setLayerInputs('fuse', {'score2', 'score_pool4'});
net.removeLayer('cropx') ;
net.setLayerInputs('fusex', {'score4', 'score_pool3'});
net.removeLayer('cropxx') ;
%% add drop layers
net.addLayer('drop1', dagnn.DropOut('rate', 0.5), 'fc6x', 'fc6_drop');
net.setLayerInputs('fc7', {'fc6_drop'});
net.addLayer('drop2', dagnn.DropOut('rate', 0.5), 'fc7x', 'fc7_drop');
net.setLayerInputs('score_fr', {'fc7_drop'});
net.addLayer('drop3', dagnn.DropOut('rate', 0.5), 'conv2_1x', 'conv2_1x_drop');
% net.setLayerInputs('conv2_2', {'conv2_1x_drop'});
%%
net.layers(1).block.size(3) = 1;
p = net.getParamIndex(net.layers(1).params{1}) ;
net.params(p).value = sum(net.params(p).value, 3);

padSizeVector = 1 * ones(1,numel(net.layers)-1);
padSizeVector([32 , 34, 36, 38 41]) = 0;     %% Specify the indices of the convolution layers that do not require padding
padSizeVector(32) = 3;

for i = 1:numel(net.layers)-1
    if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
        filt = net.getParamIndex(net.layers(i).params{1}) ;
        bias = net.getParamIndex(net.layers(i).params{2}) ;
        net.params(filt).learningRate = 0.3;
        net.params(filt).weightDecay = 0.3 ;
        net.params(bias).weightDecay = 0.3 ;
        net.params(bias).learningRate = 2 * net.params(filt).learningRate ;        
        net.layers(i).block.pad = padSizeVector(i)*ones(1,4);
        filtVal = net.params(filt).value;
        if size(filtVal,4) > 2
            net.params(filt).value(:,:,min(size(filtVal,3),60),60) = 1/size(filtVal,1) * ones(size(filtVal,1), 'single');
        end
    end
end

% pool layers pad
net.layers(5 ).block.pad = [0,0,0,0];
net.layers(10).block.pad = [0,0,0,0];
net.layers(17).block.pad = [0,0,0,0];
net.layers(24).block.pad = [0,0,0,0];
net.layers(31).block.pad = [0,0,0,0];

% Change 'score_fr' layer output. This layer is indexed as 36
for convLayerNum = [36 38 41]
    for i = [1 2]
        p = net.getParamIndex(net.layers(convLayerNum).params{i}) ;
        if i == 1
            sz = size(net.params(p).value) ;
            sz(end) = 2 ;
        else
            sz = [2 1] ;
        end
        net.params(p).value = initWeightsStd*randn(sz, 'single') ;
        net.params(p).learningRate = 2;
    end
    net.layers(convLayerNum).block.size = size(...
        net.params(net.getParamIndex(net.layers(convLayerNum).params{1})).value) ;
end

% Change 'score2' layer. This layer is indexed as 37
net.setLayerParams('score2', {'score2_filter'});
for interpLayerNum = [37 40]
    for i = 1:numel(net.layers(interpLayerNum).params)
        p = net.getParamIndex(net.layers(interpLayerNum).params{i}) ;
        if i == 1
            sz = size(net.params(p).value) ;
            sz(end-1) = 2 ;
            sz(end) = 2 ;
            filters = single(bilinear_u(4, 1, 2));
            filters = filters / sum(sum(filters(:,:,1,1)));
            net.params(p).value = filters;
        else
            sz = [] ;
            net.params(p).value = [] ;
        end
        net.params(p).learningRate = 0 ;
        net.params(p).weightDecay = 0 ;
    end
    net.layers(interpLayerNum).block.size = size(...
        net.params(net.getParamIndex(net.layers(interpLayerNum).params{1})).value) ;
    net.layers(interpLayerNum).block.hasBias = false;
    %     net.layers(interpLayerNum).block.numGroups = 2;
end
net.layers(37).block.crop = [1 1 1 1];
net.layers(40).block.crop = [1 1 1 1];

% Change 'upsample' layer. This layer is indexed as 43
for i = 1
    p = net.getParamIndex(net.layers(43).params{i}) ;
    if i == 1
        sz = size(net.params(p).value) ;
        sz(end) = 2 ;
        sz(end-1) = 2 ;
        if resizeFlag
            filters = single(bilinear_u(32, 2, 2)) ;
            net.layers(43).block.upsample = 16;
            net.layers(43).block.crop = 8*[1 1 1 1];
        else
            filters = single(bilinear_u(16, 2, 2)) ;
            net.layers(43).block.crop = 4*[1 1 1 1];
        end
        net.params(p).value = filters/sum(sum(filters(:,:,1,1)));
        net.params(p).learningRate = 0 ;
        net.params(p).weightDecay = 0;
    else
        sz = [] ;
        net.params(p).value = [];
    end
end
net.layers(43).block.size = size(...
    net.params(net.getParamIndex(net.layers(43).params{1})).value) ;
net.layers(43).block.hasBias = false;
net.layers(43).block.numGroups = 2;

net.vars(net.getVarIndex('bigscore')).precious = 0;
if resizeFlag
    net.meta.inputs.size = [256, 256, 1, 1];
    net.meta.normalization.imageSize = [256, 256, 1, 1];
else
    net.meta.inputs.size = [512, 512, 1, 1];
    net.meta.normalization.imageSize = [512, 512, 1, 1];
end

% -------------------------------------------------------------------------
% Losses and statistics
% -------------------------------------------------------------------------

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
