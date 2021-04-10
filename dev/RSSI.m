%We will have 4, 12 by 18 matrices, one for each node
%AND each matrix with a logical statement that calculates threshold based
%on inputted value
%AND all 4 logical statement matrices together
%Parse through and see which elements/locations have 1
%The corresponding coordinate on our real grid is ((x-1)/2, (y-1)/2) of the location 

RSSI_1 = csvread("Node_1.csv");
RSSI_2 = csvread("Node_2.csv");
RSSI_3 = csvread("Node_3.csv");
RSSI_4 = csvread("Node_4.csv");

data = csvread("path3_2.csv");

threshold = 4;
file_length = 13;


for c = 1:file_length
    
    N1 = data(c,2);
    N2 = data(c,3);
    N3 = data(c,4);
    N4 = data(c,5);
    
    %Calculate sparse Node matrices with 1s where threshold is met
    Node1 = (RSSI_1 < (N1 + threshold)) & (RSSI_1 > (N1 - threshold));
    Node2 = (RSSI_2 < (N2 + threshold)) & (RSSI_2 > (N2 - threshold));
    Node3 = (RSSI_3 < (N3 + threshold)) & (RSSI_3 > (N3 - threshold));
    Node4 = (RSSI_4 < (N4 + threshold)) & (RSSI_4 > (N4 - threshold));

    %Form composite matrix that shows values mostly match with measured database
    Nodes = (Node1 & Node2 & Node3 & Node4) | (Node1 & Node2 & Node3) | (Node1 & Node3 & Node4) | (Node2 & Node3 & Node4);

    numElements = 0;
    xCoordinate = 0;
    yCoordinate = 0;

    for i=1:size(Nodes,1)
        for j=1:size(Nodes,2)
            if(Nodes(i,j) == 1)
                numElements = numElements + 1;
                xCoordinate = xCoordinate + (j-1)/2;
                yCoordinate = yCoordinate + (i-1)/2;
            end 
        end 
    end 

    x = xCoordinate/numElements
    y = yCoordinate/numElements
    out(c,1) = data(c,1);
    out(c,2) = x;
    out(c,3) = y;
end    
writematrix(out, "out_path3_2.csv");
