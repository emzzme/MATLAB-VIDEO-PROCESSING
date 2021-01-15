
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.
workspace;  % workspace panel 
format long g; % format for mask
format compact; %f ormat for mask
fontSize = 15; % for figure 
tic;

checkpoint = input('Kaç noktalı doğrulama olsun?');
framenumber = input('Frame number kaça bölünsün?');

%% 2X VIDEO

videoIn = VideoReader('vid.mp4'); % video reader for video file
videoOut = VideoWriter('fot.mp4','MPEG-4'); % video writer for video file 
open(videoOut); % open video writer 
frameNum = 1; % initial frame number


while hasFrame(videoIn) % chose the odd number
    videoFrame = readFrame(videoIn); % read every frame in video
    frameNum = frameNum+1;  % frame number counter
    s = mod(frameNum,framenumber) ;    % mod selector
        if s == 0       % select odd frame
            writeVideo(videoOut,videoFrame);    % make a video file with combined every frame
        end
end
close(videoOut); % open video writer
%% FOTO Write

vid = VideoReader('fot.mp4');  % video reader for video file
  numFrames = vid.NumFrames;
  num=numFrames;
  f=0;
 
mkdir PHOTO
for i = 1:1:num
  frames = read(vid,i);
  imwrite(frames,['PHOTO\', sprintf('%03d.jpg',i)]);
    f=f+1;
end 
multi = dir('PHOTO\*.jpg*');

%% Read


% Load images.
buildingScene = imageDatastore('PHOTO\','FileExtensions',{'.jpg'});

% Display images to be stitched
montage(buildingScene.Files)

% Read the first image from the image set.
I = readimage(buildingScene, 1);

% Initialize features for I(1)
grayImage = rgb2gray(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage, points);

% Initialize all the transforms to the identity matrix. Note that the
% projective transform is used here because the building images are fairly
% close to the camera. Had the scene been captured from a further distance,
% an affine transform would suffice.
numImages = numel(buildingScene.Files);
tforms(numImages) = projective2d(eye(3));

% Initialize variable to hold image sizes.
imageSize = zeros(numImages,2);

% Iterate over remaining image pairs
for n = 2:numImages

    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;

    % Read I(n).
    I = readimage(buildingScene, n);

    % Convert image to grayscale.
    grayImage = rgb2gray(I);    

    % Save image size.
    imageSize(n,:) = size(grayImage);

    % Detect and extract SURF features for I(n).
    points = detectSURFFeatures(grayImage);    
    [features, points] = extractFeatures(grayImage, points);

    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);

    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        

    % Estimate the transformation between I(n) and I(n-1).
    
        % Estimate the transformation between I(n) and I(n-1).
    if size(indexPairs, 1) >= checkpoint
        tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    else
   % skip this frame
    end

    % Compute T(n) * T(n-1) * ... * T(1)
    tforms(n).T = tforms(n).T * tforms(n-1).T; 
end

% Compute the output limits  for each transform
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end

avgXLim = mean(xlim, 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((numel(tforms)+1)/2);

centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)    
    tforms(i).T = tforms(i).T * Tinv.T;
end

for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages

    I = readimage(buildingScene, i);   

    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);

    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)

[I2, rect] = imcrop(panorama);

disp(clock)
try
    waitforbuttonpress
    % Close figure or leave it open
    close(h)
    disp('mouse or key pressed')
catch
    disp('figure closed')
end
close all
disp(clock)

delete fot.mp4 % delete unnecessary file
[status, message, messageid] = rmdir('PHOTO', 's')

