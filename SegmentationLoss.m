classdef SegmentationLoss < dagnn.Loss

  methods
    function outputs = forward(obj, inputs, params)
      instanceWeights = zeros(size(inputs{2}));
      for imgNum = 1:size(instanceWeights,4)
          labels = inputs{2}(:,:,1,imgNum);
          backgndIdxs = double( labels == 1 );
          liverIdxs = double( labels == 2 );
          instanceWeights(:,:,1,imgNum) = instanceWeights(:,:,1,imgNum) + backgndIdxs ./ sum(backgndIdxs(:));
          instanceWeights(:,:,1,imgNum) = instanceWeights(:,:,1,imgNum) + liverIdxs ./ sum(liverIdxs(:));
      end
      outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], ...
                             'loss', obj.loss, ...
                             'instanceWeights', instanceWeights) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      instanceWeights = zeros(size(inputs{2}));
      for imgNum = 1:size(instanceWeights,4)
          labels = inputs{2}(:,:,1,imgNum);
          backgndIdxs = double( labels == 1 );
          liverIdxs = double( labels == 2 );
          instanceWeights(:,:,1,imgNum) = instanceWeights(:,:,1,imgNum) + backgndIdxs ./ sum(backgndIdxs(:));
          instanceWeights(:,:,1,imgNum) = instanceWeights(:,:,1,imgNum) + liverIdxs ./ sum(liverIdxs(:));
      end
      derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, ...
                               'loss', obj.loss, ...
                               'instanceWeights', instanceWeights) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function obj = SegmentationLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
