
dbstop if error

if exist('data\imdb.mat', 'file')
    load data\imdb.mat
else
    imdb = VocSetupLiver;
    save data\imdb imdb;
end

filterSigma = 10; 
filterSize = 7*[1 1];

GoodImages = {'0_57', '0_52', '0_67', '0_68', '0_69',  '0_71', '1_61', '1_62', '1_63', '1_70', '2_405' };
im = imread('data\Train\ct\ct_3_378.png');
imLabel = imread('data\Train\seg\seg_3_378.png');

im = im(:,:,1);
% [counts, binLoc] = imhist(im(im>0));
% figure; imhist(im(im>0))

% im = histeq(im, [0:25:255]);
% figure; imshow(imEq,[])

imFiltered = imgaussfilt(im , filterSigma, 'FilterSize', filterSize);
% imFiltered = im;

k = [1 2 1; 0 0 0; -1 -2 -1];
kk = [0 1 2;-1,  0,  1; -2, -1,  0];
H = conv2(double(imFiltered),k, 'same');
V = conv2(double(imFiltered),k','same');
U = conv2(double(imFiltered),kk,'same');
J = conv2(double(imFiltered),fliplr(kk),'same');
E = sqrt(H.*H + V.*V + U.*U + J.*J);
% medE = medfilt2(E, 5*[1 1]);
% wname = 'bior3.5';
% level = 5;
% [C,S] = wavedec2(E, level, wname);
% thr = wthrmngr('dw2ddenoLVL','penalhi',C,S,3);
% sorh = 's';
% [imDen,cfsDEN,dimCFS] = wdencmp('lvd',C,S,wname,level,thr,sorh);
% figure; subplot(1,2,1); imshow(E>80, [])
% subplot(1,2,2); imshow(imDen>80, [])
edgesIm = E > 80;
meanGrad = mean(mean(E));


% se = strel('disk', 2);
% F = imclose(edgesIm, se);
% se2nd = strel('disk', 2);
% secondClosedImg = imclose(F, se2nd);

% figure; imshow(imFiltered, []);

%-------%  Choose initial points For region-growing %-------%

thrsh = 0.085;
zerosThrs = 20000;

% Extract edges of image 
backGroundMask = ~regiongrowing(im2double(imFiltered), round(10), round(10), thrsh);
% figure; imshow(backGroundMask, []);
findZeros = sum(sum(((imFiltered == 0 & backGroundMask))));

figure; imshow(imFiltered, []);
if findZeros > zerosThrs
    x = 200;
    y= 194;
else
    [ x, y] = ginput(1);
end
hold on; plot(x,y,'r+','markersize', 10)


regGrowResult = regiongrowing(im2double(imFiltered-uint8(E/5)), round(y), round(x), thrsh);

figure; imshow(regGrowResult, [])
% negativeIm = 1- regGrowResult;

closedIm = imclose(regGrowResult , strel('disk', 3));
closedClosedImage = imclose( imclose(closedIm , strel('disk', 4)), strel('disk', 5));
% figure; imshow(closedIm, [])
figure; imshow(closedClosedImage)
figure; imshow(imLabel(:,:,1), [])
