clc;
clear all;
close all;

% Turn off all warnings
warning('off', 'all');

% Select input image
[file, path] = uigetfile('*.*', 'Select input image');
I = imread(fullfile(path, file));
figure, imshow(I);
title('Input Image');

% Convert image to grayscale if it's RGB
if size(I, 3) == 3
    I = rgb2gray(I); % Convert to grayscale using rgb2gray for better compatibility
end

% Normalize the grayscale image to the range [0, 1]
I = double(I) / 255;

% Set minimum and maximum intensity values for contrast stretching
minIntensity = 0.1;  % Adjust as needed for desired contrast
maxIntensity = 0.9;  % Adjust as needed for desired contrast

% Apply contrast stretching
I1 = (I - minIntensity) / (maxIntensity - minIntensity);
I1 = max(0, min(I1, 1)); % Clip values to the range [0, 1]
I1 = uint8(I1 * 255);  % Convert back to uint8 for display

figure, imshow(I1);
title('Enhanced Image');

% Convert the adjusted grayscale image to binary using Otsu's method
[counts, binLocations] = histcounts(I1(:), 256);
counts = counts / numel(I1); % Normalize to get probability

% Compute cumulative sums and cumulative means
cumSum = cumsum(counts);
cumMean = cumsum(counts .* binLocations(1:end-1)');

% Compute global mean
globalMean = cumMean(end);

% Compute between-class variance
sigma_b_squared = (globalMean * cumSum - cumMean).^2 ./ (cumSum .* (1 - cumSum + eps));

% Find the threshold that maximizes between-class variance
[~, maxIndex] = max(sigma_b_squared);
optimalThreshold = binLocations(maxIndex(1)); % Ensure it's a scalar

% Convert image to binary using the computed threshold
IDX = I1 > optimalThreshold; % Create binary image as logical array

% Display the binary image
figure, imshow(IDX);
title('Otsu Segmentation');

% Binary conversion
I2 = I1 > optimalThreshold; % Convert to binary (logical) format based on threshold
I2 = uint8(I2) * 255; % Convert logical to uint8 for display

figure, imshow(I2);
title('Binary Image');

% Thinning using the custom function
thin_image = thinImage(I2); % Use the custom thinning function instead of bwmorph
figure; imshow(thin_image, []); title('Thinned Image'); % Ensure proper display with [] to scale correctly

% Define the custom thinImage function
function thinnedImage = thinImage(binaryImage)
    thinnedImage = double(binaryImage); % Convert to double for processing
    while true
        % Erode the image using a custom erosion function
        erodedImage = customErode(thinnedImage);
        
        % Subtract the eroded image from the original
        tempImage = thinnedImage - erodedImage;
        
        % Update the thinned image
        thinnedImage = thinnedImage - tempImage;
        
        % Check if there are any changes
        if all(tempImage(:) == 0) % If no changes, exit loop
            break;
        end
    end
    thinnedImage = uint8(thinnedImage); % Convert back to uint8 for display
end

% Custom erosion function to replace imerode
function output = customErode(inputImage)
    [rows, cols] = size(inputImage);
    output = zeros(rows, cols);
    % Apply erosion using a 3x3 neighborhood
    for i = 2:rows-1
        for j = 2:cols-1
            % Check if all pixels in the 3x3 neighborhood are white
            if all(all(inputImage(i-1:i+1, j-1:j+1) == 255))
                output(i, j) = 255; % Erode if condition met
            end
        end
    end
end

% Minutiae extraction from the thinned image
if isempty(thin_image)
    error('Thinned image is empty. Check the thinning process.');
end

% Extract minutiae features (e.g., ridge endings and bifurcations)
[ridge_endings, bifurcations] = extractMinutiae(thin_image);
outImg = labelMinutiae(thin_image, ridge_endings, bifurcations);  % Label minutiae
figure; imshow(outImg, []); title('Minutiae'); % Ensure the image scales properly

% Function to extract minutiae (ridge endings and bifurcations)
function [ridge_endings, bifurcations] = extractMinutiae(thinnedImage)
    ridge_endings = [];
    bifurcations = [];
    
    [rows, cols] = size(thinnedImage);
    for x = 2:rows-1
        for y = 2:cols-1
            neighborhood = thinnedImage(x-1:x+1, y-1:y+1);
            num_pixels = sum(neighborhood(:) == 255); % Count white pixels
            
            if thinnedImage(x, y) == 255
                if num_pixels == 2
                    ridge_endings = [ridge_endings; x, y]; % Store ridge ending
                elseif num_pixels == 3
                    bifurcations = [bifurcations; x, y]; % Store bifurcation
                end
            end
        end
    end
end

% Function to label and display minutiae (ridge endings and bifurcations)
function labeledImg = labelMinutiae(thinnedImage, ridge_endings, bifurcations)
    labeledImg = thinnedImage;  % Start with the thinned image

    % Label ridge endings (marked as white crosses)
    for i = 1:size(ridge_endings, 1)
        labeledImg(ridge_endings(i, 1), ridge_endings(i, 2)) = 255;  % Mark as white
        % Add cross for ridge ending
        labeledImg(ridge_endings(i, 1)-1:ridge_endings(i, 1)+1, ridge_endings(i, 2)-1:ridge_endings(i, 2)+1) = 255;
    end

    % Label bifurcations (marked as white circles)
    for i = 1:size(bifurcations, 1)
        labeledImg(bifurcations(i, 1), bifurcations(i, 2)) = 255;  % Mark as white
        % Add circle for bifurcation
        labeledImg(bifurcations(i, 1)-1:bifurcations(i, 1)+1, bifurcations(i, 2)-1:bifurcations(i, 2)+1) = 255;
    end
end

% Load dataset and feature extraction
matlabrootPath = 'C:\Users\gani2\OneDrive\Documents\Desktop\TK11664\TK11664\main program';
digitDatasetPath = fullfile(matlabrootPath, 'datasets');
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Prepare features and labels from the dataset
features = [];
labels = [];
for i = 1:length(imds.Files)
    img = readimage(imds, i);
    if size(img, 3) == 3
        img = rgb2gray(img); % Ensure image is grayscale
    end
    img = double(img) / 255; % Normalize
    img = img > 0.5; % Example processing step
    feature_vector = [mean(img(:)), std(img(:))]; % Feature extraction
    features = [features; feature_vector]; % Collect features
    labels = [labels; imds.Labels(i)]; % Collect corresponding labels
end

% Training data analysis
figure;
hold on;
unique_labels = unique(labels);
colors = lines(numel(unique_labels)); % Generate colors
for i = 1:numel(unique_labels)
    scatter(features(labels == unique_labels(i), 1), features(labels == unique_labels(i), 2), 36, colors(i, :), 'filled');
end
xlabel('Mean Intensity');
ylabel('Standard Deviation');
title('Training Data Analysis');
grid on;

% Manual KNN Classification
k = 3; % Number of neighbors

% Check if we have more features than neighbors
if size(features, 1) < k
    k = size(features, 1); % Adjust k to the number of features
end

% Predict label for the input image
predictedLabel = manualKNN(features, labels, feature_vector, k);

% Display the classification results
msgbox(['Predicted label: ', char(predictedLabel)]);

% Testing data analysis
testFeatures = feature_vector; % Using the same feature vector for testing in this example
figure;
hold on;
% Plot training data
for i = 1:numel(unique_labels)
    scatter(features(labels == unique_labels(i), 1), features(labels == unique_labels(i), 2), 36, colors(i, :), 'filled');
end
% Plot test feature
plot(testFeatures(1), testFeatures(2), 'kx', 'MarkerSize', 10, 'LineWidth', 2); % Plot test feature
xlabel('Mean Intensity');
ylabel('Standard Deviation');
title('Testing Data Analysis');
legend('Class 1', 'Class 2', 'Test Sample');
grid on;

% Manual KNN Function
function label = manualKNN(featuresTrain, labelsTrain, testPoint, k)
    if k > size(featuresTrain, 1)
        k = size(featuresTrain, 1); % Adjust k to the number of samples
    end
    
    distances = sqrt(sum((featuresTrain - testPoint).^2, 2));
    [~, sortedIndices] = sort(distances);
    nearestIndices = sortedIndices(1:k);
    nearestLabels = labelsTrain(nearestIndices);
    label = mode(nearestLabels);
end
