%% Get Data
%collectData;
%load('C:\Users\Rafael\Box Sync\CS341\Classification\encoding2.mat')
%load('C:\Users\Rafael\Box Sync\CS341\Classification\encodinghop.mat')
%load('C:\Users\Rafael\Box Sync\CS341\Classification\encodingNorma.mat')
load('C:\Users\Rafael\Box Sync\CS341\Classification\datanorm.mat')
%% Weighting
S = 9;
weights = [1 1 .01 .01 .4 .3 .6 .1 .01 1e3];
%weights = [1 1 .00 .00 .4 .3 .6 .0 .00 0e3];
encoding(isnan(encoding)) = 0;

% Add Weights to the feature vector
for i = 1:S
    encoding(i,:) = encoding(i,:)/max(abs(encoding(i,:)))*weights(i);
%    encoding(i,:) = encoding(i,:)/std(encoding(i,:))*weights(i);
end

encoding(1:S,:) = encoding(1:S,:)/S;
encoding((S+1):end,:) = encoding((S+1):end,:)*...
    weights(10)/(size(encoding,1)-S);

M = 219+833;
N = [100 120 140 180 220 240 250];
K = 10;
NORM = 2;
trials = 10;
FN = zeros(trials,size(N,2));
FP = zeros(trials,size(N,2));

for t = 1:trials
    for j = 7
        NN = N(j);
        squiT = encoding(:,randperm(833,NN));
        nonT = encoding(:,M+randperm(7438,10*NN));

    %   c1 = sum(squiT,2);
        c1 = median(squiT,2); % Create squiggle cluster

        [c2, idx, E] = vl_kmeans(nonT,K,...
        'initialization','plusplus',...
        'distance', ['l' num2str(NORM)],...
        'Algorithm','Lloyd',...
        'MaxNumIterations',1000);

        for kk = 1:K
           c2(:,kk) =  median(nonT(:,idx == kk),2);     %other clusters    
        end

        % Take random selection
        squi = encoding(:,randperm(833,2*NN));
        non = encoding(:,M+randperm(7438,20*NN));

        % Find false negatives
        for i = 1:(2*NN)
            ss = squi(:,i);
            norms = [norm(ss-c1,NORM) norm(ss-c2(:,1),NORM)...
                norm(ss-c2(:,2),NORM) norm(ss-c2(:,3),NORM)...
                norm(ss-c2(:,4),NORM) norm(ss-c2(:,5),NORM)...
                norm(ss-c2(:,6),NORM) norm(ss-c2(:,7),NORM)...
                norm(ss-c2(:,8),NORM) norm(ss-c2(:,9),NORM)...
                norm(ss-c2(:,10),NORM)];
            [a b] = min(norms);
            b;
            if(b ~= 1)
                FN(t,j) = FN(t,j)+1;
            end
        end
        FN(t,j) = FN(t,j)/(2*NN);

        % Find false positives
        for i = 1:(20*NN)
            ss = non(:,i);
            norms = [norm(ss-c1,NORM) norm(ss-c2(:,1),NORM)...
                norm(ss-c2(:,2),NORM) norm(ss-c2(:,3),NORM)...
                norm(ss-c2(:,4),NORM) norm(ss-c2(:,5),NORM)...
                norm(ss-c2(:,6),NORM)];
            [a b] = min(norms);
            if(b == 1)
                FP(t,j) = FP(t,j)+1;
            end
        end
        FP(t,j) = FP(t,j)/(20*NN);
    end
    t
    FP
    FN
end
FP = mean(FP);
FN = mean(FN);

plot(N,FP*100,N,FN*100,'linewidth',2)
ylim([0 10])
ylabel('Error Rate (%)')
xlabel('Training Data Size')
legend('False Positive','False Negative')
set(gca,'linewidth', 2);
set(gca,'fontsize', 27);