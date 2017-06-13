cd('C:\Users\dcorc\OneDrive\TAU 2\Advanced Topics in Medical Image Processing 1\CNN_project\matconvnet-fcn')
addpath(genpath(fullfile(pwd)))
addpath(genpath('C:\Program Files\MATLAB\MatConvNet'))
% vl_compilenn('enableGpu', true, 'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\', 'cudaMethod', 'nvcc', 'enableCudnn', true, 'cudnnRoot', 'C:\Program Files\MATLAB\MatConvNet\cuda', 'verbose', 1)
vl_setupnn
% vl_testnn('gpu', true)

net16 = load('pascal-fcn16s-dag.mat');
net16 = dagnn.DagNN.loadobj(net16);

imdb = VocSetupLiver;
save data\imdb.mat  imdb
