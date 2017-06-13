function outputSize = CalcOutputSize(inputSize, weightsSize, pad, stride)

xySizes = ( inputSize(1:2)+pad(1:2) - weightsSize(1:2)+1 ) ./ stride;
outputSize = [xySizes, weightsSize(4)];
