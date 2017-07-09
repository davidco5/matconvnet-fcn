dbstop if error

if exist('savedImages.mat', 'file')
    load('savedImages.mat');
end

TrainDirPath = 'data\Test\ct\';
destinationDirPath = 'data\Test\seg - regGrow\';
cTestImages = dir('data\Test\ct\*.png');

filterSigma = 10; 
filterSize = 7*[1 1];

isQuantize = false;

if ~exist('currIm', 'var')
    currIm = 443;
    initialSeedX = 95;
    initalSeedY = 264;
end
currIm = currIm - 1;
while currIm < numel(cTestImages)
    currIm = currIm+1
    close all;
    
    origImagePath = fullfile(TrainDirPath, cTestImages(currIm).name);
    im = imread(origImagePath);
    im = im(:,:,1);
    
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
    
    figure;
    title (fullfile('Image name: ',cTestImages(currIm).name), 'fontsize',12); 
    subplot(1,2,1)
    imshow(imMinusGrad, []);
    title(fullfile('Image minus Gradient ( ', cTestImages(currIm).name, ')' ),'fontsize',12);
    subplot(1,2,2)
    imshow(closedClosedImage);
    title('Segmentationt','fontsize',12);
    
    questResponse = MFquestdlg([0.35 0.3], 'What would you like to do next?', 'Segmentation result menu', 'Save segmentation' , 'Repeat with Quantization', 'Add another region','Pause run');
    
    switch(questResponse)
        case 'Save segmentation'
            destImagePath = fullfile(destinationDirPath, cTestImages(currIm).name);
            imwrite(closedClosedImage, destImagePath); 
            isQuantize = false; 
            savedImages{currIm} = cTestImages(currIm).name;
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
            
            advancedCoice = MFquestdlg([0.35 0.3], 'What would you like to do?' ,'Two regions menu', 'Save segmentation', 'Pause run');
            
            switch(advancedCoice)
                case 'Save segmentation'
                    destImagePath = fullfile(destinationDirPath, cTestImages(currIm).name);
                    imwrite(totalImageAfterClose, destImagePath); 
                    isQuantize = false;
                    savedImages{currIm} = cTestImages(currIm).name;
                    continue;
                case 'Pause run'
                    pause(inf);
            end
            
            case 'Pause run'
                    pause(inf);
          
    end

end

save ('savedImages.mat', 'savedImages');
            