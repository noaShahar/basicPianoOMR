function [k_to_scores, nearestLine,is_right,i ] = fillScoresIndexes( i,y_loc, right_hand_lines, is_right,nearestLine ,staffLocations,w,score_i,k_to_scores)    
    for j=1:length(right_hand_lines)
           limit = right_hand_lines(j,:);
           if (y_loc >= limit(1) && y_loc <= limit(2))
               is_right(i) = 1;
               hamsha = staffLocations(staffLocations >=limit(1) & staffLocations<= limit(2));
               extented_hamsha = [hamsha(1)-3*w, hamsha(1)-2*w ,hamsha(1)-1*w, hamsha, hamsha(end)+1*w,hamsha(end)+2*w ,hamsha(end)+3*w];
               mid_hamsha = (extented_hamsha(:,1:end-1) + extented_hamsha(:,2:end)) /2;
               [~,ind] = min(abs([extented_hamsha, mid_hamsha]-y_loc));
               isALine =1; % 1 - on the lines
               if ind >=11
                   ind = ind -11;
                   isALine = 0; % 0 - between the lines 
               end 
               if (isempty(ind))
                   throw error;
               end
               nearestLine(i,:) = [ind, isALine];
               k_to_scores(i)=score_i;
               i = i+1;
               break;
           end
    end
end

