
dbstop if error

if exist('data\imdb.mat', 'file')
    load data\imdb.mat
else
    imdb = VocSetupLiver;
    save data\imdb imdb;
end

filterSigma = 10; 
filterSize = 7*[1 1];

GoodImages = ['ct_0_57' 'ct_0_52' 'ct_0_67' 'ct_0_68' 'ct_0_69'  'ct_0_71' 'ct_1_61' 'ct_1_62' 'ct_1_63' 'ct_1_70' 'ct_2_405' ];
im = imread('C:\Users\Admin\Desktop\DeepLearningProject\data\Train\ct\ct_3_468.png');
imLabel = imread('C:\Users\Admin\Desktop\DeepLearningProject\data\Train\seg\seg_3_468.png');

im = im(:,:,1);

% [counts, binLoc] = imhist(im(im>0));
% figure; imhist(im(im>0))

% im = histeq(im, [0:25:255]);
% figure; imshow(imEq,[])

imFiltered = imgaussfilt(im , filterSigma, 'FilterSize', filterSize);
% imFiltered = im;

% k = [1 2 1; 0 0 0; -1 -2 -1];
% kk = [0 1 2;-1,  0,  1; -2, -1,  0];
% H = conv2(double(imFiltered),k, 'same');
% V = conv2(double(imFiltered),k','same');
% U = conv2(double(imFiltered),kk,'same');
% J = conv2(double(imFiltered),fliplr(kk),'same');
% E = sqrt(H.*H + V.*V + U.*U + J.*J);
% 
% medE = medfilt2(E, 5*[1 1]);
% edgesIm = medE > 90;
% 
% 
% se = strel('disk', 2);
% F = imclose(edgesIm, se);
% se2nd = strel('disk', 2);
% secondClosedImg = imclose(F, se2nd);

figure; imshow(imFiltered, []);

%-------%  Choose initial points For region-growing %-------%

thrsh = 0.085;
zerosThrs = 20000;

% Extract edges of image 
backGroundMask = 1- regiongrowing(im2double(imFiltered), round(10), round(10), thrsh);
% figure; imshow(backGroundMask, []);
findZeros = sum(sum(((imFiltered == 0 & backGroundMask))));

if findZeros > zerosThrs
    x = 205;
    y= 194;
else
    [ x, y] = ginput(1);
end;
    
figure; imshow(imFiltered, []);
% [ x, y] = ginput(1);


regGrowResult = regiongrowing(im2double(imFiltered), round(y), round(x), thrsh);

figure; imshow(regGrowResult, [])
negativeIm = 1- regGrowResult;

closedIm = imclose(negativeIm , strel('disk', 3));
closedClosedImage = imclose( imclose(closedIm , strel('disk', 4)), strel('disk', 5));
figure; imshow(closedIm, [])
figure; imshow(closedClosedImage)

figure; imshow(imLabel(:,:,1), [])
