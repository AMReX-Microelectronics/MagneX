data = load('Mdata2000.txt');

[X,Y] = meshgrid(1:166,1:42);

My_in = data(:,5);

My = reshape(My_in,166,42);

figure
surf(X, Y, My.')
shading interp
colorbar
set(gca,'Ydir','reverse')
title('My')