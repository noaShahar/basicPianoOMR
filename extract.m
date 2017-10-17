function  X = extract(d ,obj_size)
    files = dir(d);
    i=1;
    X = [];
    for k = 3:length(files)
        entry = strcat(d,'\',files(k).name);
        if isdir(entry)
            [Xe] = extract(entry);
            X = [X ; Xe];
            i = i + size(Xe,1);
        else
            signIm = imread(entry);
            if size(signIm, 3)==3
                signIm = rgb2gray(signIm);
            end
            signIm = im2bw(signIm);
            signIm = im2double(signIm);
            signIm = imresize(signIm,obj_size);
            X(i,:) = signIm(:);
            i = i+1;
            if mod(i,1000) == 0
                disp(i)
            end
        end

    end
end

