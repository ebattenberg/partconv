%% select an audio clip and a reverb

audio_path = '../audio/';
reverb_path = '../reverbs/';

songs = dir([audio_path '*.wav']);
reverbs = dir([reverb_path '*.wav']);

disp('Audio clips')
for i = 1:length(songs)
    fprintf('%u. %s\n',i,songs(i).name);
end
SONG = input('Select a song number: ');
fprintf('\n');

disp('Reverbs')
for i = 1:length(reverbs)
    fprintf('%u. %s\n',i,reverbs(i).name);
end
REVERB = input('Select a reverb number: ');
fprintf('\n');


%  read in data and run convolution

NN = [0 14];
FS = 44100;


%[x,xfs] = wavread([audio_path songs(SONG).name],...
 %   [1+NN(1)*FS NN(2)*FS]);
%x = x(:,1);
% [x,xfs] = wavread([audio_path songs(SONG).name]);

% [h,hfs] = wavread('./reverbs/Church Schellingwoude.wav');
[h,hfs] = wavread([reverb_path reverbs(REVERB).name]);
h = h(:,1); %only use one channel for now
h = h/norm(h);

% if xfs ~= FS || hfs ~= FS
%     error('sampling frequencies do not agree');
% end

% % custom partitioning
% p = struct;
% n = 256;
% p.N = [n 8*n];  %size of partitions at each level
% p.numN = [8 3];     %number of partitions at each level
% 
% pout = p;

L = length(h);




% optimal partitioning

n = 64; % initial blocksize
tic
[pout, opt_cost] = optimal_part_load(L,n);
toc
%%
p = pout;

temp = p.N(1)*p.numN(1);
% h(1:temp) = 0;

h = single(h);
x = single(x);
%overlap-add Matlab function for comparison.
tic
y_add = fftfilt(h,x);
t = toc;
fprintf('\nfftfilt time: %g\n',t);



tic
[y_save, pout] = MFDL_part_conv(x,h,p);
% [y_save, pout] = FDL_part_conv(x,h,n);
t = toc;
fprintf('part_conv time: %g\n',t);


cost = part_conv_work(pout);
fprintf('work = %i per output sample\n',cost);
fprintf('work coef = %g (ms/work unit)\n',t/cost*1000);

nh = length(h);
nx = length(x);

N = p.N;

y1 = y_add(1:nx);
y2 = y_save(N(1)+1:nx+N(1)); %adjust for latency

E = sum((y1-y2).^2)/sum(y1.^2);



fprintf('filter length = %g (%.2g sec)\n',nh,nh/FS);
fprintf('signal length = %g (%.2g sec)\n',nx,nx/FS);
fprintf('latency (N) = %g (%.3g ms)\n\n',N(1),1000*N(1)/FS);
if E > eps
    fprintf('  ERROR = %g\n\n',E);
end

figure(1); clf;
subplot(2,1,1)
plot([y1 y2])
subplot(2,1,2)
plot(y1-y2)
title('error')
figure(2); clf;
visualize_partitions(pout,h)




%% listen to original
soundsc(x,FS);

%% listen to processed
soundsc(y_save,FS);


