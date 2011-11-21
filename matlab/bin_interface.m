


%% read in binary matrix hmat.bin

path = '../C/';
fid = fopen([path 'y_1d_out.bin'],'r');


dim = fread(fid,2,'int');
mat = fread(fid,dim(1)*dim(2),'float');
hmat = reshape(mat,dim(2),dim(1))';
plot(hmat');

fclose(fid);

%% read in wav
% N = floor(2^15*1.4);
N = 1;  
cout = wavread('../C/output.wav');
cout = cout(N:end);
cout = cout(1:length(y_add));

subplot(2,1,1)
plot([y_add cout])
subplot(2,1,2)
plot(y_add - cout);

% plot(y_add - 4*cout(1:length(y_add)))

%%
soundsc(cout,44100);

%%
soundsc(y_add,44100);





%% read in vec
N = 256;
N = N + 1; 
file = '../C/y_1d_out.bin';
% file = '../C/wav_out.bin';
fid = fopen(file,'r');


dim = fread(fid,1,'int');
vec = fread(fid,dim,'float');
fclose(fid);


target = y_add;
vec = 2*vec(N:N-1+length(y_add));

figure(1)
subplot(2,1,1)
plot([vec target]);
subplot(2,1,2)
plot(vec-target);

disp(norm(vec-target)/norm(target))


%%
%% read in vec
N = 2048;
N = N + 1; 
file = '../C/output.wav';
% file = '../C/wav_out.bin';
vec = wavread(file);


target = y_add;
vec = 2*vec(N:N-1+length(y_add));
vec = vec/mean(vec)*mean(target);

figure(1)
subplot(2,1,1)
plot([vec target]);
subplot(2,1,2)
plot(vec-target);

disp(norm(vec-target)/norm(target))

