function [data1_interp,data2_interp,t1_interp,t2_interp] = interpolation2dat(data1,data2,t1,t2)

% Min length

len_data1 = length(data1);
len_data2 = length(data2);

min_length = min(len_data1,len_data2);

% Interpolation cases

if min_length == len_data1
    data1_interp = data1;
    t1_interp = t1;
    data2_interp = interp1(t2,data2,t1);
    t2_interp = t1;
    
elseif min_length == len_data2
    data1_interp = interp1(t1,data1,t2);
    t1_interp = t2;
    data2_interp = data2;
    t2_interp = t2;
end
