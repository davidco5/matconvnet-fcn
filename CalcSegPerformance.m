function sSegStats = CalcSegPerformance(predictMat, seg)

TP = sum( predictMat(:) & seg(:) );
FP = sum( predictMat(:) & ~seg(:) );
FN = sum( ~predictMat(:) & seg(:) );
sSegStats.Sens = TP / (TP + FN);
sSegStats.PPV = TP / (TP + FP);
sSegStats.Dice = 2*TP / (2*TP + FP + FN);
sSegStats.TP = TP;
sSegStats.FP = FP;
sSegStats.FN = FN;