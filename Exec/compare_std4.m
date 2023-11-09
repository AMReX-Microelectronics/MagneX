nx = 166; % number of cells on x direction
ny = 42;
nz = 1;

dx = 3; % cell size on x direction, in nanometers
dy = 3;
dz = 3;

dt = 5E-6; % nanoseconds
timesteps = 50000; % hack 150000;
alpha = 0.5; % damping constant
exchConstant = 1.3E-11 * 1E18; % nanometer/nanosecond units
% exchConstant = 1E-13 * 1E18; % nanometer/nanosecond units
mu_0 = 1.256636; % = 4 * pi / 10
Ms = 800; % saturation magnetization
Mx = repmat(Ms, [nx ny nz]);
% Mx = repmat(Ms/sqrt(3), [nx ny nz]); % magnetization on x direction 
My = zeros([nx ny nz]); % TODO: reduce truncation and zeropadding into one step
Mz = My;

Kxx = zeros(nx * 2, ny * 2, nz * 2); % Initialization of demagnetization tensor
Kxy = Kxx;
Kxz = Kxx;
Kyy = Kxx;
Kyz = Kxx;
Kzz = Kxx;
prefactor = 1 / 4 / 3.14159265;
for K = -nz + 1 : nz - 1 % Calculation of Demag tensor, see NAKATANI JJAP 1989
    for J = -ny + 1 : ny - 1
        for I = -nx + 1 : nx - 1
            if I == 0 && J == 0 && K == 0
                continue
            end
            L = I + nx; % shift the indices, b/c no negative index allowed in MATLAB
            M = J + ny;
            N = K + nz;
            for i = 0 : 1 % helper indices
                for j = 0 : 1
                    for k = 0 : 1
                        r = sqrt ( (I+i-0.5)*(I+i-0.5)*dx*dx... 
                            +(J+j-0.5)*(J+j-0.5)*dy*dy...
                            +(K+k-0.5)*(K+k-0.5)*dz*dz);
                        
                        Kxx(L,M,N) = Kxx(L,M,N) ...
                            + (-1).^(i+j+k) * atan( ...
                            (K+k-0.5) * (J+j-0.5) * dz * dy / r / (I+i-0.5) / dx);
                        
                        Kxy(L,M,N) = Kxy(L,M,N) ...
                            + (-1).^(i+j+k) ...
                            * log( (K+k-0.5) * dz + r);
                        
                        Kxz(L,M,N) = Kxz(L,M,N) ...
                            + (-1).^(i+j+k) ...
                            * log( (J+j-0.5) * dy + r);
                        
                        Kyy(L,M,N) = Kyy(L,M,N) ...
                            + (-1).^(i+j+k) * atan( ...
                            (I+i-0.5) * (K+k-0.5) * dx * dz / r / (J+j-0.5) / dy);
                        
                        Kyz(L,M,N) = Kyz(L,M,N) ...
                            + (-1).^(i+j+k) ...
                            * log( (I+i-0.5) * dx + r);
                        
                        Kzz(L,M,N) = Kzz(L,M,N) ...
                            + (-1).^(i+j+k) * atan( ...
                            (J+j-0.5) * (I+i-0.5) * dy * dx / r / (K+k-0.5) / dz);
                    end
                end
            end
            Kxx(L,M,N) = Kxx(L,M,N) * prefactor;
            Kxy(L,M,N) = Kxy(L,M,N) * - prefactor;
            Kxz(L,M,N) = Kxz(L,M,N) * - prefactor;
            Kyy(L,M,N) = Kyy(L,M,N) * prefactor;
            Kyz(L,M,N) = Kyz(L,M,N) * - prefactor;
            Kzz(L,M,N) = Kzz(L,M,N) * prefactor;
%             fprintf('%d %d %d %f %f %f %f %f %f \n', ...
%                 L - nx, M - ny, N - nz, Kxx(L,M,N), Kxy(L,M,N), Kxz(L,M,N), ...
%                 Kyy(L,M,N), Kyz(L,M,N), Kzz(L,M,N));
        end
    end
end % calculation of demag tensor done
% Kxx = permute (Kxx, [3 2 1]);

Kxx_fft = fftn(Kxx); % fast fourier transform of demag tensor
Kxy_fft = fftn(Kxy); % need to be done only once
Kxz_fft = fftn(Kxz);
Kyy_fft = fftn(Kyy);
Kyz_fft = fftn(Kyz);
Kzz_fft = fftn(Kzz);

for t = 1 : timesteps
    Mx(end + nx, end + ny, end + nz) = 0; % zero padding
    My(end + nx, end + ny, end + nz) = 0;
    Mz(end + nx, end + ny, end + nz) = 0;
        
    Hx = ifftn(fftn(Mx) .* Kxx_fft + fftn(My) .* Kxy_fft + fftn(Mz) .* Kxz_fft); % calc demag field with fft
    Hy = ifftn(fftn(Mx) .* Kxy_fft + fftn(My) .* Kyy_fft + fftn(Mz) .* Kyz_fft);
    Hz = ifftn(fftn(Mx) .* Kxz_fft + fftn(My) .* Kyz_fft + fftn(Mz) .* Kzz_fft);

    Hx = Hx (nx:(2 * nx - 1), ny:(2 * ny - 1), nz:(2 * nz - 1) ); % truncation of demag field
    Hy = Hy (nx:(2 * nx - 1), ny:(2 * ny - 1), nz:(2 * nz - 1) );
    Hz = Hz (nx:(2 * nx - 1), ny:(2 * ny - 1), nz:(2 * nz - 1) );

    if mod(t, 1000) == 0
        outFileH = fopen(['Hdata',num2str(t),'.txt'], 'w');
        for k = 1 : nz
        for j = 1 : ny
        for i = 1 : nx
            fprintf(outFileH, '%d\t%d\t%d\t%f\t%f\t%f\n', ...
                    i, j, k, Hx(i,j,k), Hy(i,j,k), Hz(i,j,k));
        end
        end
        end
        fclose(outFileH)
    endif

    Mx = Mx (1:nx, 1:ny, 1:nz); % truncation of Mx, remove zero padding
    My = My (1:nx, 1:ny, 1:nz);
    Mz = Mz (1:nx, 1:ny, 1:nz);
    
    exch = 2 * exchConstant / mu_0 / Ms / Ms;
    for i = 1 : nx
        for j = 1 : ny
            for k = 1 : nz
                Hx(i,j,k) = Hx(i,j,k) + exch / dx / dx * ( ...
                    Mx(i-1 + ~(i-1),j,k) + Mx(i+1 - ~(i-nx),j,k) ...
                    + Mx(i,j-1 + ~(j-1),k) + Mx(i,j+1 - ~(j-ny),k) ...
                    + Mx(i,j,k-1 + ~(k-1)) + Mx(i,j,k+1 - ~(k-nz)) ...
                    - 6 * Mx(i,j,k) ...
                    );
                Hy(i,j,k) = Hy(i,j,k) + exch / dy / dy * ( ...
                    My(i-1 + ~(i-1),j,k) + My(i+1 - ~(i-nx),j,k) ...
                    + My(i,j-1 + ~(j-1),k) + My(i,j+1 - ~(j-ny),k) ...
                    + My(i,j,k-1 + ~(k-1)) + My(i,j,k+1 - ~(k-nz)) ...
                    - 6 * My(i,j,k) ...
                    );
                Hz(i,j,k) = Hz(i,j,k) + exch / dz / dz * ( ...
                    Mz(i-1 + ~(i-1),j,k) + Mz(i+1 - ~(i-nx),j,k) ...
                    + Mz(i,j-1 + ~(j-1),k) + Mz(i,j+1 - ~(j-ny),k) ...
                    + Mz(i,j,k-1 + ~(k-1)) + Mz(i,j,k+1 - ~(k-nz)) ...
                    - 6 * Mz(i,j,k) ...
                    );
            end
        end
    end

    % hbias in
    if t < 4000 % timestep 4000 = 0.02 s
        Hx = Hx + 100; % multiply by 1,000 in MagneX
        Hy = Hy + 100;
        Hz = Hz + 100;
    elseif t < 6000 % timestep 6000 = 0.03 s
        Hx = Hx + (6000 - t) / 20; % multiply by 1,000 in MagneX
        Hy = Hy + (6000 - t) / 20;
        Hz = Hz + (6000 - t) / 20;
    elseif t > 50000 % timestep 50000 = 0.25 s
        % mu_0 Hx = -24.6 mT -> Hx = 19.576
        % MagneX should use 19,576 [A/m]
        % mu_0 Hy = 4.3 mT
        % and by the same logic, Hy = 3,422 [A/m]
        Hx = Hx - 19.576;
        Hy = Hy + 3.422;
        alpha = 0.02;
    end
    
    MxHx = My .* Hz - Mz .* Hy;
    MxHy = Mz .* Hx - Mx .* Hz;
    MxHz = Mx .* Hy - My .* Hx;
    
    prefactor1 = (-0.221) * dt / (1 + alpha * alpha);
    prefactor2 = prefactor1 * alpha / Ms;
    deltaMx = prefactor1 .* MxHx + prefactor2 .* (My .* MxHz - Mz .* MxHy);
    deltaMy = prefactor1 .* MxHy + prefactor2 .* (Mz .* MxHx - Mx .* MxHz);
    deltaMz = prefactor1 .* MxHz + prefactor2 .* (Mx .* MxHy - My .* MxHx);
    
    Mx = Mx + deltaMx;
    My = My + deltaMy;
    Mz = Mz + deltaMz;
    
    mag = sqrt(Mx .* Mx + My .* My + Mz .* Mz);
    Mx = Mx ./ mag * Ms;
    My = My ./ mag * Ms;
    Mz = Mz ./ mag * Ms;
    
    if mod(t, 1000) == 0
        MxMean = mean(mean(Mx));
        MyMean = mean(mean(My));
        MzMean = mean(mean(Mz));
        snapFile = fopen(['Mvstime',num2str(t),'.txt'], 'w');
        fprintf(snapFile, '%d\t%f\t%f\t%f\r\n', t, MxMean/Ms, MyMean/Ms, MzMean/Ms);
        fclose(snapFile)

        outFileM = fopen(['Mdata',num2str(t),'.txt'], 'w');
        for k = 1 : nz
            for j = 1 : ny
                for i = 1 : nx
                    fprintf(outFileM, '%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\r\n', ... 
                        i, j, k, Mx(i,j,k)/Ms, My(i,j,k)/Ms, Mz(i,j,k)/Ms, ...
                        Hx(i,j,k), Hy(i,j,k), Hz(i,j,k));
                end
            end
        end
        fclose(outFileM);
    end
    fprintf('Done with step %d\n',t)
end
fclose('all');

%% plot
%load('Mvstime.txt')
%load('std4_magtense.txt')
%close all

%figure
% plot(Mvstime(25:end,1), Mvstime(25:end,2),'b');
%hold on
%plot(Mvstime(25:end,1)*dt-50000*dt, Mvstime(25:end,3),':r','LineWidth',2);
% plot(Mvstime(25:end,1), Mvstime(25:end,4),'--k');
%plot(std4_magtense(:,1)*1e9, std4_magtense(:,3),'--b','LineWidth',1);
%legend('My_{matlab}','My_{magtense}')
% legend('Mx','My','Mz')