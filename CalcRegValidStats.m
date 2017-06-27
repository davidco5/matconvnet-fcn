mySegPath = 'data\Val\seg - regGrow\';
origSegPath = 'data\Val\seg\';
cTestImages = dir([mySegPath, '*.png']);
sSegStats = struct('TP', [], 'FP', [], 'FN', [], 'Sens', [], 'PPV', [], 'Dice', []);
sSegStats = repmat(sSegStats, 1, numel(cTestImages));

for imNum = 1:numel(cTestImages)
    myImagePath = fullfile(mySegPath, cTestImages(imNum).name);
    mySeg = imread(myImagePath);
    mySeg = logical(mySeg(:,:,1));
    origImagePath = fullfile(origSegPath, strrep(cTestImages(imNum).name, 'ct', 'seg'));
	origSeg = imread(origImagePath);
    origSeg = logical(origSeg(:,:,1));
    sSegStats(imNum) = CalcSegPerformance(mySeg, origSeg);
    fprintf('%d\n', imNum)
end
meanSens = mean([sSegStats.Sens])
meanPPV = mean([sSegStats.PPV])
meanDice = mean([sSegStats.Dice])

save data\regSegStats.mat sSegStats