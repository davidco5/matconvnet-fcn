dbstop if error

% load data\sSegStatsReg.mat

TrainDirPath = 'data\val\ct\';
originalSegDirPath = 'data\val\seg\';
destinationDirPath = 'data\val\seg - regGrow\';
cValImages = dir('data\val\ct\*.png');

sSegStats = struct('TP', [], 'FP', [], 'FN', [], 'Sens', [], 'PPV', [], 'Dice', []);
sSegStats = repmat(sSegStats, 1, numel(cValImages));

filterSigma = 10; 
filterSize = 7*[1 1];

isQuantize = false;

if ~exist('currIm', 'var')
    currIm = 64;
    initialSeedX = 95;
    initalSeedY = 264;
end
currIm = currIm - 1;
while currIm < numel(cValImages)
    currIm = currIm+1
    close all;
    
    origImagePath = fullfile(TrainDirPath, cValImages(currIm).name);
    segPath = fullfile(originalSegDirPath, strrep(cValImages(currIm).name, 'ct', 'seg'));
    im = imread(origImagePath);
    im = im(:,:,1);
    seg = imread(segPath);
    seg = seg(:,:,1);
    
    imFiltered = imgaussfilt(im , filterSigma, 'FilterSize', filterSize);
    
    k = [1 2 1; 0 0 0; -1 -2 -1];
    kk = [0 1 2;-1,  0,  1; -2, -1,  0];
    H = conv2(double(imFiltered),k, 'same');
    V = conv2(double(imFiltered),k','same');
    U = conv2(double(imFiltered),kk,'same');
    J = conv2(double(imFiltered),fliplr(kk),'same');
    E = sqrt(H.*H + V.*V + U.*U + J.*J);
    
    % medE = medfilt2(E, 5*[1 1]);
    edgesIm = E > 80;
    meanGrad = mean(mean(E));
    
    thrsh = 0.08;
    
    imMinusGrad = imFiltered-uint8(E/1.3);
    
    if (isQuantize)
        QntThrsh = multithresh(imMinusGrad, 10);
        imQuantized = imquantize(imMinusGrad, QntThrsh);
        imQnt = uint8(imQuantized);
        imMinusGrad = imQnt;
        isQuantize = false;
        thrsh = 0.006;
    end
	figure; imshow(imMinusGrad, []);  

    if (imFiltered(initalSeedY, initialSeedX )== 0 )
        [ initialSeedX , initalSeedY] = ginput(1);
       
    else
        [ seedX , seedY] = ginput(1);
%         seedX = initialSeedX;
%         seedY = initalSeedY;
    end
    
    regGrowResult = regiongrowing(im2double(imMinusGrad), round(seedY), round(seedX), thrsh);
    
    closedIm = imclose(regGrowResult , strel('disk', 8));
    closedClosedImage = imclose( imclose(closedIm , strel('disk', 4)), strel('disk', 5));
    sSegStats(currIm) = CalcSegPerformance(closedClosedImage, seg);
    
    figure;
    title (fullfile('Image name: ',cValImages(currIm).name), 'fontsize',12); 
    subplot(1,2,1)
    imshow(imMinusGrad, []);
    title(fullfile('Image minus Gradient ( ', cValImages(currIm).name, ')' ),'fontsize',10);
    subplot(1,2,2)
    imshow(closedClosedImage);
%     title('Segmentationt','fontsize',12);
    title(sprintf('Sens = %.2f, PPV = %.2f, Dice = %.2f', sSegStats(currIm).Sens, sSegStats(currIm).PPV, sSegStats(currIm).Dice), 'fontsize', 12)
    
    questResponse = MFquestdlg([0.35 0.3], 'What would you like to do next?', 'Segmentation result menu', 'Save segmentation' , 'Repeat with Quantization', 'Add another region','Pause run');
    
    switch(questResponse)
        case 'Save segmentation'
            destImagePath = fullfile(destinationDirPath, cValImages(currIm).name);
            imwrite(uint8(closedClosedImage)*255, destImagePath); 
            isQuantize = false; 
            savedImages{currIm} = cValImages(currIm).name;
            continue;
        case 'Repeat with Quantization'
            currIm = currIm - 1;
            isQuantize = true;
            continue;
        case 'Add another region'
            isQuantize = false;
            firstSeg = closedClosedImage;
            
            figure; imshow(imMinusGrad, []);
            [ x, y] = ginput(1);
            initialSeedX = round(x);
            initialSeedY = round(y);
            
            regGrowResultSecond = regiongrowing(im2double(imMinusGrad), round(y), round(x), thrsh);
            
            totalImg = or(firstSeg,regGrowResultSecond);
            
            totalImageAfterClose = imclose( totalImg, strel('disk', 5));
            figure; imshow(totalImageAfterClose, [])
            sSegStats(currIm) = CalcSegPerformance(totalImageAfterClose, seg);
            title(sprintf('Sens = %.2f, PPV = %.2f, Dice = %.2f', sSegStats(currIm).Sens, sSegStats(currIm).PPV, sSegStats(currIm).Dice), 'fontsize', 12)
            
            advancedCoice = MFquestdlg([0.35 0.3], 'What would you like to do?' ,'Two regions menu', 'Save segmentation', 'Pause run');
            
            switch(advancedCoice)
                case 'Save segmentation'
                    destImagePath = fullfile(destinationDirPath, cValImages(currIm).name);
                    imwrite(uint8(totalImageAfterClose)*255, destImagePath); 
                    isQuantize = false;
                    savedImages{currIm} = cValImages(currIm).name;
                    continue;
                case 'Pause run'
                    pause(inf);
            end
            
            case 'Pause run'
                    pause(inf);
          
    end

end

save data\sSegStatsReg.mat sSegStats
