classdef SegmentationLoss < dagnn.Loss

  methods
    function outputs = forward(obj, inputs, params)
      mass = sum(sum(inputs{2} > 0,2),1) + 1 ;
      instanceWeights = inputs{2};
      backgndIdxs = instanceWeights == 1;
      liverIdxs = instanceWeights == 2;
      nLiverPix = sum(sum(liverIdxs,2),1);
      for imgNum = 1:size(numel(nLiverPix))
          instanceWeights(backgndIdxs(:,:,1,imgNum)) = 1 ./ (mass(imgNum) - nLiverPix(imgNum));
          instanceWeights(liverIdxs(:,:,1,imgNum)) = 1 ./ nLiverPix(imgNum);
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
      mass = sum(sum(inputs{2} > 0,2),1) + 1 ;
      derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, ...
                               'loss', obj.loss, ...
                               'instanceWeights', 1./mass) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function obj = SegmentationLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
