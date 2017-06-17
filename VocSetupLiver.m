function imdb = VocSetupLiver(varargin)
% opts.edition = '07' ;
opts.dataDir = fullfile('data') ;
opts.archiveDir = fullfile('data','archives') ;
opts.includeDetection = false ;
opts.includeSegmentation = true ;
opts.includeTest = false ;
opts = vl_argparse(opts, varargin) ;

% % Download data
% if ~exist(fullfile(opts.dataDir,'Annotations')), download(opts) ; end

% Source images and classes
cFoldNames = {'ct', 'seg'};
imdb.paths.image.train = esc( fullfile(opts.dataDir, 'Train', 'ct', '%s.png') );
imdb.paths.image.val = esc( fullfile(opts.dataDir, 'Val', 'ct', '%s.png') );
imdb.paths.image.test = esc( fullfile(opts.dataDir, 'Test', 'ct', '%s.png') );
imdb.paths.segmentation.train = esc( fullfile(opts.dataDir, 'Train', 'seg', '%s.png') );
imdb.paths.segmentation.val = esc( fullfile(opts.dataDir, 'Val', 'seg', '%s.png') );
imdb.paths.segmentation.test = esc( fullfile(opts.dataDir, 'Test', 'seg', '%s.png') );
imdb.sets.id = uint8([1 2 3]) ;
imdb.sets.name = {'train', 'val', 'test'} ;
imdb.classes.id = uint8(1:2) ;
imdb.classes.name = {'NotLiver', 'Liver'} ;
imdb.classes.images = cell(1,2) ;
imdb.images.id = [] ;
imdb.images.name = {} ;
imdb.images.set = [] ;
imdb = addImageSet(opts, imdb, 'train', 1, cFoldNames) ;
imdb = addImageSet(opts, imdb, 'val', 2, cFoldNames) ;
if opts.includeTest, imdb = addImageSet(opts, imdb, 'test', 3, cFoldNames) ; end

% Compress data types
imdb.images.id = uint32(imdb.images.id) ;
imdb.images.set = uint8(imdb.images.set) ;

function imdb = addImageSet(opts, imdb, setName, setCode, cFoldNames)
% -------------------------------------------------------------------------
j = length(imdb.images.id) ;
cImageSet = dir(fullfile(opts.dataDir, setName, cFoldNames{1}, '*.png'));
for i=1:length(cImageSet)
    imdb.images.id(j+i) = j+i;
    imdb.images.set(j+i) = setCode ;
    imdb.images.name{j+i} = strrep( strrep(cImageSet(i).name, cFoldNames{1}, ''), '.png', '') ;
    imdb.images.classification(i+j) = false ;
    if exist(fullfile(opts.dataDir, setName, cFoldNames{2}, [cFoldNames{2}, imdb.images.name{j+i}, '.png']), 'file')
        imdb.images.segmentation(j+i) = true ;
    else
        imdb.images.segmentation(j+i) = false ;
    end
    info = imfinfo(fullfile(opts.dataDir, setName, cFoldNames{1}, cImageSet(i).name));
    imdb.images.size(:,i+j) = uint16([info.Width ; info.Height]) ;
end

function str=esc(str)
% -------------------------------------------------------------------------
str = strrep(str, '\', '\\') ;
