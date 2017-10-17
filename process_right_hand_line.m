function linesLocations = process_right_hand_line(start,fin,im,w)
%PROCESS a right hand line
%%
thershold = 0.9;
%%
%%%%%%%%%%%%%%%%% looking for teivot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
partial_im = im(start:fin,:);
%sumVec = sum(partial_im,1)';
sumVec = sum ( partial_im > graythresh(im) , 1)';
debugSumVec = sumVec;
linesLocations = [];
initialMax = max(sumVec);
curMax = initialMax;
index =1;
while curMax > thershold*initialMax
   [~, centerLoc] = max(debugSumVec);
   left = max ([centerLoc - 0.5*w,0]);
   right = min ([centerLoc + 0.5*w,size(debugSumVec,1)]);
   linesLocations(index)= centerLoc;
   debugSumVec(left:right) = 0;
   index  = index + 1;
   curMax = max(debugSumVec);
end
plot(sumVec);
hold on;
scatter(linesLocations, sumVec(linesLocations), 50, 'r', 'filled');
hold off;
figure;
imshow(partial_im,[])
end