%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Cleaning workspace %%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc
%%
files = dir('scores');
for file=3:length(files)
    
    resizeFactor = 4;
    imageName = strcat('scores\',files(file).name);
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
    distanceMax = size(im,1)/5/5;
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% create a sumVec  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    BW= im2bw(im,graythresh(im));
    BW = im2double(BW);
    sumVec = sum(BW,2);
    % figure;
    % plot(sumVec);
    % oneLineFilter = ones(1,size(im,2));
    % im = imfilter(im, oneLineFilter,'replicate');
    % mid = ceil(size(im,2)/2);
    % sumVec = im(:,mid);
    figure;
    plot(sumVec);
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
    
    margin = ceil(0.2*w);
    line_margins = ceil(0.1*w-1);

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
    figure;
    plot(vecStaff);
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
    figure(1);
    plot(sumVec);
    hold on;
    scatter(staffLocations, sumVec(staffLocations), 50, 'r', 'filled');
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
    imNoLines = imadjust(im);

    %% show detected lines
    im_r = imNoLines;
    im_g = imNoLines;
    im_b = imNoLines;

    for i=1:length(staffLocations)
        im_r(max(staffLocations(i)-line_margins-3,1) : min(staffLocations(i)+line_margins+3, size(im,1)),:) = 0;
        im_g(max(staffLocations(i)-line_margins-3,1) : min(staffLocations(i)+line_margins+3, size(im,1)),:) = 0;
    end
    

    im_res = cat(3, im_r, im_g, im_b);
    imshow(im_res,[]);
    %%
    im_gray_all{file} = im;
    sum_vec_all{file} = sumVec;
    im_colored_all{file,1} = im_res;
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% auto complete line objects %%%%%%%%%%%%%%%%
    for i=1:length(staffLocations)
        imNoLines(staffLocations(i),:)=0;
        up_u = min (staffLocations(i)+line_margins+margin, size(imNoLines,1));
        up_d = staffLocations(i)+line_margins;
        down_d = max(staffLocations(i)-line_margins -margin,1);
        down_u = staffLocations(i)-line_margins;
        partialIm = imNoLines([down_d:down_u up_d:up_u],:);
        partialIm = imgaussfilt(partialIm,0.75);
        %sumLines = sum ( partialIm > graythresh(im) , 1);
        sumLines = sum(partialIm,1);
        compLine = sumLines > 0.3 * max(sumLines);
        for j=-line_margins-3:line_margins+3
            if (staffLocations(i)+j >=1 && staffLocations(i)+j < size(im,1))
                imNoLines(staffLocations(i)+j,:)= im(staffLocations(i)+j,:).*compLine;
            end
        end
        %imNoLines(staffLocations(i)-5 : staffLocations(i)+5 ,:) = repmat(im(staffLocations(i),:).*compLine,11,1);
    end
    figure(file)
    imshow(imNoLines,[])
    file


end

