%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Cleaning workspace %%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Constants %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
resizeFactor = 4;
imageName = 'LONDON_BRIDGE-page-001.jpg';
distanceMin = 10;
thershold = 0.6;
%numOfStaffs = 8;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Read the image to im %%%%%%%%%%%%%%%%%%%%%%%%%
im = imread(imageName);
im = rgb2gray(im);
im = im2double(im);
im = imresize(im,resizeFactor);
im = imcomplement(im);
figure;
imshow(im,[]);
distanceMax = size(im,1)/9;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% create a sumVec  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BW= im2bw(im,graythresh(im));
BW = im2double(BW);
sumVec = sum(BW,2);
figure;
plot(sumVec);
% oneLineFilter = ones(1,size(im,2));
% im = imfilter(im, oneLineFilter,'replicate');
% mid = ceil(size(im,2)/2);
% sumVec = im(:,mid);
% figure;
% plot(sumVec);
%%
%%%%%%%%%%%%%%%%%%%%%% Find the staff spacing - w %%%%%%%%%%%%%%%%%%%%%%%%%
tic
close all
%distanceMin = 10;
%distanceMax = 100;
distance_vec = distanceMin:distanceMax;
mat = zeros(size(im,1), length(distance_vec));
for i = 1:length(distance_vec)
    distance = distance_vec(i);
    z = zeros(1,distance);
    staff = [z, repmat([1, z],1,5)]';
    newCol = imfilter(sumVec,staff,'replicate');
    mat(:,i) = newCol;
%     = [mat,newCol];
end
t = toc
% 
% for distance = 1:5 %distanceMin : distanceMax
%     z = zeros(1,distance);
%     staff = [z, repmat([1, z],1,5)]';
%     figure;
%     plot(sumVec);
%     newCol = conv(sumVec,staff,'same');
%     figure;
%     plot(newCol);    
%     mat = [mat,newCol];
% end
% stop
% figure;
% imshow(mat,[]);

% figure;
% plot(highestValuesArray);
highestValuesArray = max(mat,[],1);
[~,w_idx] = max(highestValuesArray);
w = distance_vec(w_idx);
% figure;
% imshow(im,[]);
% figure; plot(mat(:,w_idx))
%%
%%%%%%%%%%%%%% Creating the staff filter based on spacing w %%%%%%%%%%%%%%%
% this is the convo lution of im sumVec and staff with the correct w
% z = zeros(1,w);
% staff = [z, repmat([1, z],1,5)]';
% vecStaff = imfilter(sumVec,staff,'replicate');
vecStaff = mat(:,w_idx);
% figure;
% plot(vecStaff);
% stop
    %%
    %%%%%%%%%%%%%%%%%%%%% Extracting the staffs locations %%%%%%%%%%%%%%%%%%%%%
    % staff locations will be in staffLocations[]
    org_vecStaff = mat(:,w_idx);
    staffLocations = [];
    initialMax = max(vecStaff);
    curMax = initialMax;
    index =0;
    while curMax > thershold*initialMax  %prctile(vecStaff,97) %100 - (8*5*5/length(vecStaff)));
       [~, centerLoc] = max(vecStaff);
       left = max ([centerLoc - 3*w,1]);
       right = min ([centerLoc + 3*w,size(vecStaff,1)]);
       [pointsY,pointsX] = findpeaks(sumVec(left:right));
    %    [~, s_i] = sort(pointsY, 'descend');
    %    pointsX(s_i);
       sorted = [pointsX+left-1 ,pointsY];
       added_lines = zeros(1,5);
       for i=1:5
           sorted = sortrows([sorted(:,1),sorted(:,2)],2);
           if (isempty(sorted))
               throw( MException('Noa:Tal','shouldnt happen'));
           end
           current_line =  sorted(end,1);
           added_lines(i) = current_line;
           left_1 = max (current_line - line_margins - margin,1);
           right_1 = min (current_line + line_margins + margin, length(vecStaff));
           sorted = sorted(sorted(:,1)< left_1 | sorted(:,1)>right_1,:);
       end
       if (length(added_lines) < 5)
           break;
       end
       staffLocations(5*index+1:5*index+5)= added_lines;
       vecStaff(left:right) = 0;
%        figure;
%        plot(vecStaff);
%        hold on;
%        scatter(staffLocations,org_vecStaff(staffLocations),50,'r','filled')

       index  = index + 1;
       curMax = max(vecStaff);
    end
    % staffLocations = sort(staffLocations);
%%
%%%%%%%%%%%%%%%%%%%% Prestent the extracted staff %%%%%%%%%%%%%%%%%%%%%%%%%

% presented = zeros(size(sumVec,1),1);
% presented(staffLocations) = max(sumVec);
% presented(staffLocations) = sumVec(staffLocations);
% for i =1 :  size(staffLocations,2)
%     presented(staffLocations(i),1) = max(sumVec);
% end 
% figure;
% hold on
% plot(sumVec);
% plot(presented);
close all
figure;
plot(sumVec);
hold on;
scatter(staffLocations, sumVec(staffLocations), 50, 'r', 'filled');
hold off

%%
%%%%%%%%%%%%%%%%%%%%%%%% find tievot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
staffLocations = sort(staffLocations);
starts = staffLocations(1:5:end)-3*w;
ends= staffLocations(5:5:end)+3*w;
right_hand_lines = [starts(1:2:end)',ends(1:2:end)'];
left_hand_lines = [starts(2:2:end)',ends(2:2:end)'];
for i=1:length(right_hand_lines)
%    teivot = process_right_hand_line(right_hand_lines(i,1),left_hand_lines(i,2),im,w);
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%% notes detection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% remove lines %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
imNoLines = imadjust(im);
margin = ceil(0.2*w);
line_margins = ceil(0.1*w-1);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% auto complete line objects %%%%%%%%%%%%%%%%
for i=1:length(staffLocations)
    imNoLines(staffLocations(i),:)=0;
    up_u = staffLocations(i)+line_margins+margin;
    up_d = staffLocations(i)+line_margins;
    down_d = staffLocations(i)-line_margins -margin;
    down_u = staffLocations(i)-line_margins;
    partialIm = imNoLines([down_d:down_u up_d:up_u],:);
    partialIm = imgaussfilt(partialIm,0.75);
    %sumLines = sum ( partialIm > graythresh(im) , 1);
    sumLines = sum(partialIm,1);
    compLine = sumLines > 0.3 * max(sumLines);
    for j=-line_margins-3:line_margins+3
        imNoLines(staffLocations(i)+j,:)= im(staffLocations(i)+j,:).*compLine;
    end
    %imNoLines(staffLocations(i)-5 : staffLocations(i)+5 ,:) = repmat(im(staffLocations(i),:).*compLine,11,1);
end
%figure;imshow(imNoLines,[]);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% blob detection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% trials
im2 = imNoLines;
stop
im2 = imgaussfilt(imNoLines);
%% create a note filter
% trial 1
filter_size = ceil(2*w);
r = ceil(0.3*w);
filter_circle = zeros(filter_size);
for x=1:filter_size
    for y=1:filter_size
        if (w-x)*(w-x) + (w-y)*(w-y)<= r*r && (w-x)*(w-x) + (w-y)*(w-y)>= (r-line_margins)*(r - line_margins)
            filter_circle(x,y)=1;
        end
    end
end
imshow(filter_circle,[]);
%% trial 2 - gil's filer
load('half_note_filter.mat');
half_note_filter = H;
Rfactor = 116/w;
imresized = imresize(imNoLines,Rfactor);
filtered = filter2(half_note_filter,imresized);
maximas = imregionalmax(filtered);
potenial_notes = filtered > max(filtered(:))/2 ;%prctile(filtered(find(filtered)),95);
cents = maximas & potenial_notes;
figure; imshow(filter2(half_note_filter,cents),[]);
figure; imshow(im2,[]);
[Ycenter,Xcenter] = ind2sub(size(cents),find(cents)); % where cents is 1 (find(cents)), give the 2D index
centers = [Xcenter ,Ycenter]/Rfactor;
%%
 filtered = conv2(imNoLines,filter_circle); figure; imshow(filtered);
%filtered = imNoLines(2116:2171,2278:2357); % empty note
%filtered = imNoLines(1605:1690,1590:1670); % full note
imresized = imresize(imNoLines,1/resizeFactor);
filtered = imresize(filtered,1/(1.1*resizeFactor));
filtered = im2bw(filtered,graythresh(imresized));
BW = im2bw(imresized,graythresh(imresized));
SE = strel('arbitrary',filtered);
filtered_erode = imerode(BW,SE);
imshow(filtered_erode,[]);
%% original code
im2 = imNoLines;
im2 = imgaussfilt(imNoLines);

centers = [];
radii = [];
for i=1:length(right_hand_lines)
    paritalIm = im2(right_hand_lines(i,1):left_hand_lines(i,2),:);
    [centers1, radii1, metric] = imfindcircles(paritalIm,[ceil(0.4*w) w],'Sensitivity',0.87);%0.955);
    centers = [centers ; centers1 + repmat([0 right_hand_lines(i,1)],length(centers1),1)];
    radii = [radii ; radii1];
end

[centersNew,radiiNew] = RemoveOverLap(centers,radii,pi*w/8,2);
figure;
imshow(im2);
viscircles(centersNew, radiiNew,'EdgeColor','b');

%% %%%%%%%%%%%%%%%%%%%%%%%%%% dataset preparation %%%%%%%%%%%%%%%%%%%%%%%%% 
obj_size= [80 30];

Xnotes = extract('dataset\Note',obj_size);
Xsym = extract('dataset\Symbol',obj_size);


X = [Xnotes; Xsym];
Y = [ones(size(Xnotes,1),1); zeros(size(Xsym,1),1)];

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% knn %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Mdl = fitcknn(X,Y,'NumNeighbors',3,'Standardize',1);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% blob recognition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
imshow(im2,[]);
scores = [];
for i=1:length(centersNew)
   center = centersNew(i,:);
   radius = 2*radiiNew(i);
   rect = [center(1)-radius,center(2)-radius,2*radius,4*radius];
   object = imcrop(im2,rect);
   object = imresize(object,obj_size);
   input = object(:)';
   classResult = predict(Mdl,input);
   if classResult==1
       color = 'b';
       scores = [scores; centersNew(i,:)];
   else 
       color = 'r';
   end
   rectangle('Position',rect,'EdgeColor', color);
   
   %object =
   %im2(center(1)-radius:center(1)+radius,center(2)-radius:center(2)+radius);
 end
%% show detected scores
figure;
imshow(im2,[]);
scores = centers;
for i=1:length(scores)
    r = 0.5*w;
    center = scores(i,:);
    rect = [center(1)-r,center(2)-r,2*r,2*r];
    rectangle('Position',rect,'EdgeColor', 'b');
end
%%
%%% trying the KNN %%%
signIm = imread('test2_symbol.jpg');
if size(signIm, 3)==3
    signIm = rgb2gray(signIm);
end
signIm = im2double(signIm);
signIm = imresize(signIm,obj_size);
signIm = imcomplement(signIm);
input = signIm(:)';

classResult = predict(Mdl,input)

%% sound detection - scores blob centers are in "scores"
%% identify which of the centers belongs to each hand
scores = sortrows(scores,1);
is_right = zeros(length(scores),1);
is_left = zeros(length(scores),1);
nearestLine = zeros(length(scores),2);
num_of_scores = length(scores);
k_to_scores = zeros(num_of_scores,1);

% initialize matrix:
N = num_of_scores;
M = zeros(N,6);

k = 1; % this is i of s_right, is_left, nearest line, according to the sequence of the scores.

for j=1:length(right_hand_lines) % length(right_hand_lines) = num of שורות in the files
    for i=1:num_of_scores
    y_loc = scores(i,2);
    if  (y_loc >=right_hand_lines(j,1) && y_loc<=left_hand_lines(j,2))
        [k_to_scores, nearestLine,is_right,k ] = fillScoresIndexes(k,y_loc, right_hand_lines, is_right,nearestLine ,staffLocations,w,i,k_to_scores);
        [k_to_scores, nearestLine,is_left,k ] = fillScoresIndexes(k,y_loc, left_hand_lines, is_left,nearestLine ,staffLocations,w,i,k_to_scores);
    end
    end 
end
%% prepare midi
% prepare map from our detection to midi numbers
right_map_is_line =  [88 84 81 77 74 71 67 64 60 57 53 ];
right_map_is_not_aLine = [86 83 79 76 72 69 65 62 59 55 ];

left_map_is_line =  [67 64 60 57 53 50 47 43 40 36 33 ];
left_map_is_not_aLine = [65 62 59 55 52 48 45 41 38 35];

midi_nums = zeros(num_of_scores,1);
for i=1: num_of_scores
    if (is_right(i))
        if (nearestLine(i,2)==1)
            map = right_map_is_line;
        else
            map = right_map_is_not_aLine;
        end
    elseif (is_left(i))
        if (nearestLine(i,2)==1)
            map = left_map_is_line;
        else
            map = left_map_is_not_aLine;
        end

    end
    midi_nums(i) = map(nearestLine(i,1));
end
%% calc time
t  =0;
need_to_fill_prev = false;
M(1,5) = t;
for k=2:num_of_scores
    prev_score = k_to_scores(k-1);
    curr_score = k_to_scores(k);
    if (abs(scores(prev_score,1) - scores(curr_score,1))<0.25*w)
        M(k,5)=t;
        need_to_fill_prev =true;
    else
        t =t+0.5;
        M(k,5)=t;
        M(k-1,6)=t;
        if (need_to_fill_prev)
            M(k-2,6)=t;
        end
    end
end

%%
% initialize matrix:

M(:,1) = 1;         % all in track 1
M(:,2) = 1;         % all in channel 1
M(:,3) = midi_nums; % note numbers
M(:,4) = 120;       %  volumes
% M(:,5) = (0:.5:num_of_scores/2) M(:,6) = M(:,5)+0.5;


midi_new = matrix2midi(M);
writemidi(midi_new, 'londonBridge.mid');


