nx = 32; % number of cells on x direction
ny = 32;
nz = 8;

dx = 1.171875e-9;
dy = 1.09375e-9;
dz = 6.25e-10;

Ms = 8.e5; % saturation magnetization

Mx = zeros([nx ny nz]); % TODO: reduce truncation and zeropadding into one step
My = Mx;
Mz = Mx;

% restrict My to subregion
for k = 3 : 6
for j = 3 : 30
for i = 4 : 29
  My(i,j,k) = Ms;          
end
end
end

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
        end
    end
end % calculation of demag tensor done

outFile = fopen('Kdata.txt', 'w');

for k = 1 : 2*nz
    for j = 1 : 2*ny
        for i = 1 : 2*nx
            fprintf(outFile, '%d\t%d\t%d\t%f\t%f\t%f\n', ...
                    i, j, k, Kyy(i,j,k), Kyy(i,j,k), Kyy(i,j,k));
        end
    end
end

fprintf(outFile,'\r\n');

Kxx_fft = fftn(Kxx); % fast fourier transform of demag tensor
Kxy_fft = fftn(Kxy); % need to be done only once
Kxz_fft = fftn(Kxz);
Kyy_fft = fftn(Kyy);
Kyz_fft = fftn(Kyz);
Kzz_fft = fftn(Kzz);

outFile = fopen('Kyyfftdata.txt', 'w');

for k = 1 : 2*nz
    for j = 1 : 2*ny
        for i = 1 : 2*nx
            fprintf(outFile, '%d\t%d\t%d\t%f\t%f\t%f\n', ...
                    i, j, k, real(Kyy_fft(i,j,k)), imag(Kyy_fft(i,j,k)), imag(Kyy_fft(i,j,k)));
        end
    end
end
fprintf(outFile,'\r\n');

outFile = fopen('Kyzfftdata.txt', 'w');

for k = 1 : 2*nz
    for j = 1 : 2*ny
        for i = 1 : 2*nx
            fprintf(outFile, '%d\t%d\t%d\t%f\t%f\t%f\n', ...
                    i, j, k, real(Kyz_fft(i,j,k)), imag(Kyz_fft(i,j,k)), imag(Kyz_fft(i,j,k)));
        end
    end
end
fprintf(outFile,'\r\n');

outFile = fopen('Hdata.txt', 'w');

Mx(end + nx, end + ny, end + nz) = 0; % zero padding
My(end + nx, end + ny, end + nz) = 0;
Mz(end + nx, end + ny, end + nz) = 0;

Hx = ifftn(fftn(Mx) .* Kxx_fft + fftn(My) .* Kxy_fft + fftn(Mz) .* Kxz_fft); % calc demag field with fft
Hy = ifftn(fftn(Mx) .* Kxy_fft + fftn(My) .* Kyy_fft + fftn(Mz) .* Kyz_fft);
Hz = ifftn(fftn(Mx) .* Kxz_fft + fftn(My) .* Kyz_fft + fftn(Mz) .* Kzz_fft);

Hx = Hx (nx:(2 * nx - 1), ny:(2 * ny - 1), nz:(2 * nz - 1) ); % truncation of demag field
Hy = Hy (nx:(2 * nx - 1), ny:(2 * ny - 1), nz:(2 * nz - 1) );
Hz = Hz (nx:(2 * nx - 1), ny:(2 * ny - 1), nz:(2 * nz - 1) );

for k = 1 : nz
    for j = 1 : ny
        for i = 1 : nx
            fprintf(outFile, '%d\t%d\t%d\t%f\t%f\t%f\n', ...
                    i, j, k, Hx(i,j,k), Hy(i,j,k), Hz(i,j,k));
        end
    end
end

fprintf(outFile,'\r\n');

fclose('all');