
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.
workspace;  % workspace panel 
format long g; % format for mask
format compact; %f ormat for mask
fontSize = 15; % for figure 
tic;


[status, message, messageid] = rmdir('PHOTO', 's');

fold = input('Kaç frame aralıkla inceleme yapılsın?   ');
fol = fold;

%% FOTO Write

vid = VideoReader('2.mp4');  % video reader for video file
numFrames = vid.NumFrames;
num=numFrames;
q=1;
u=101;
w=num/fold;
  
mkdir PHOTO

for f = 1:1:w
    
    mkdir (['PHOTO\' int2str(u) '\' int2str(f) '\']); % alt klsör oluşturma
    
    for i = q:1:fold  % fotolaro sıralı yazma
        frames = read(vid,i);
        imwrite(frames,['PHOTO\' int2str(u) '\' int2str(f) '\', sprintf('%03d.jpg',i)]);
    end
    
    if fol*f < num-fol  % fotoğraf numaralarını seçerek for başlangıcını belirleme
        q=q+fol;
        fold=fold+fol;
    else
        fold = num;
    end
end

clearvars -except w u o fol num f
close all;


checkpoint = input('Kaç noktalı doğrulama olsun?   ');

%% Read
q=1;
h = 1;
d = f;
true = 0;
targetSize = [1069 1900];

while true == 0
    if d > fol
        d = d / fol;
        h = h + 1;
    else
        true = 1;
        h = h+1;
    end
end    
%%
for m = 1:1:h
    
    u = u+1;
    b = 1;
    p = 1;
    mkdir (['PHOTO\' int2str(u) '\' int2str(p) '\']); % alt klsör oluşturma

    for j = 1:1:f
        % Load images.

        buildingScene = imageDatastore(['PHOTO\' int2str(u-1) '\' int2str(j) '\'],'FileExtensions',{'.jpg'});
        
        if u > 103
            targetSize = [1013 1800];
        end

        % Display images to be stitched
        %montage(buildingScene.Files)

        % Read the first image from the image set.
        I = readimage(buildingScene, 1);
        
        r = centerCropWindow2d(size(I),targetSize);
        m1 = imcrop(I,r);

        % Initialize features for I(1)
        grayImage = rgb2gray(m1);
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

        %%clear
        clearvars -except tforms imageSize I numel numImages buildingScene checkpoint fol num u p b f h j targetSize
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
        if j > 1
            if mod(j,fol) == 1
                p = p+1;
                mkdir (['PHOTO\' int2str(u) '\' int2str(p) '\']); % alt klsör oluşturma
            end
        end
        
        C = panorama;
        r = centerCropWindow2d(size(C),targetSize);
        m2 = imcrop(C,r);
        

        imwrite(m2,fullfile(['PHOTO\' int2str(u) '\' int2str(p) '\', sprintf('%02d.jpg',j)]),'jpg');
        b = b+1;
    end
    if b > fol
        f = b / fol;
    else
        f = b;
    end
    j = 1;
end

%[status, message, messageid] = rmdir('PHOTO', 's');
