nRuns = 50;
blockSizes = 100 * ones(1, 7); % Up to [700 x 700] matrices

% Dim 2 has 2x2 product of matrix type and inversion method.
timesRandom = zeros(length(blockSizes), 4, nRuns);
timesBlockDiags = zeros(length(blockSizes), 4, nRuns);
timesBlockFulls = zeros(length(blockSizes), 4, nRuns);

%% Test 1: full matrices (built-in inv)

for iBlock=1:length(blockSizes)
    for iRun=1:nRuns
        MBlockDiag = randBlockDiags(blockSizes(1:iBlock), true);
        MBlockFull = randFullBlocks(blockSizes(1:iBlock), true);
        MRand = randn(size(MBlockDiag));
        
        tstart = tic;
        [~] = inv(MRand);
        timesRandom(iBlock, 1, iRun) = toc(tstart);
        
        tstart = tic;
        [~] = inv(MBlockDiag);
        timesBlockDiags(iBlock, 1, iRun) = toc(tstart);
        
        tstart = tic;
        [~] = inv(MBlockFull);
        timesBlockFulls(iBlock, 1, iRun) = toc(tstart);
    end
end

%% Test 2: sparse matrices (built-in inv)

for iBlock=1:length(blockSizes)
    for iRun=1:nRuns
        MBlockDiag = randBlockDiags(blockSizes(1:iBlock), false);
        MBlockFull = randFullBlocks(blockSizes(1:iBlock), false);
        MRand = randn(size(MBlockDiag));
        
        tstart = tic;
        [~] = inv(MRand);
        timesRandom(iBlock, 2, iRun) = toc(tstart);
        
        tstart = tic;
        [~] = inv(MBlockDiag);
        timesBlockDiags(iBlock, 2, iRun) = toc(tstart);
        
        tstart = tic;
        [~] = inv(MBlockFull);
        timesBlockFulls(iBlock, 2, iRun) = toc(tstart);
    end
end

%% Test 3: full matrices (blockwise inv)

for iBlock=1:length(blockSizes)
    for iRun=1:nRuns
        MBlockDiag = randBlockDiags(blockSizes(1:iBlock), true);
        MBlockFull = randFullBlocks(blockSizes(1:iBlock), true);
        MRand = randn(size(MBlockDiag));
        
        tstart = tic;
        [~] = blockinv(MRand, blockSizes(1:iBlock));
        timesRandom(iBlock, 3, iRun) = toc(tstart);
        
        tstart = tic;
        [~] = blockinv(MBlockDiag, blockSizes(1:iBlock));
        timesBlockDiags(iBlock, 3, iRun) = toc(tstart);
        
        tstart = tic;
        [~] = blockinv(MBlockFull, blockSizes(1:iBlock));
        timesBlockFulls(iBlock, 3, iRun) = toc(tstart);
    end
end

%% Test 4: sparse matrices (blockwise inv)

for iBlock=1:length(blockSizes)
    for iRun=1:nRuns
        MBlockDiag = randBlockDiags(blockSizes(1:iBlock), false);
        MBlockFull = randFullBlocks(blockSizes(1:iBlock), false);
        MRand = randn(size(MBlockDiag));
        
        tstart = tic;
        [~] = blockinv(MRand, blockSizes(1:iBlock));
        timesRandom(iBlock, 4, iRun) = toc(tstart);
        
        tstart = tic;
        [~] = blockinv(MBlockDiag, blockSizes(1:iBlock));
        timesBlockDiags(iBlock, 4, iRun) = toc(tstart);
        
        tstart = tic;
        [~] = blockinv(MBlockFull, blockSizes(1:iBlock));
        timesBlockFulls(iBlock, 4, iRun) = toc(tstart);
    end
end

%% Aggregate

meanTimesRandom = mean(timesRandom, 3);
meanTimesBlockDiag = mean(timesBlockDiags, 3);
meanTimesBlockFull = mean(timesBlockFulls, 3);

%% Plots

subplot(2,2,1);
hold on;
plot(cumsum(blockSizes), meanTimesRandom(:, 1));
plot(cumsum(blockSizes), meanTimesBlockDiag(:, 1));
plot(cumsum(blockSizes), meanTimesBlockFull(:, 1));
set(gca, 'YScale', 'log');
legend('randn()', 'block-diags', 'block-full', 'Location', 'southeast');
xlabel('Matrix Width');
ylabel('Average Compute Time');
title('Built-In inv, ''full'' type');

subplot(2,2,2);
hold on;
plot(cumsum(blockSizes), meanTimesRandom(:, 2));
plot(cumsum(blockSizes), meanTimesBlockDiag(:, 2));
plot(cumsum(blockSizes), meanTimesBlockFull(:, 2));
set(gca, 'YScale', 'log');
legend('randn()', 'block-diags', 'block-full', 'Location', 'southeast');
xlabel('Matrix Width');
ylabel('Average Compute Time');
title('Built-In inv, ''sparse'' type');

subplot(2,2,3);
hold on;
plot(cumsum(blockSizes), meanTimesRandom(:, 3));
plot(cumsum(blockSizes), meanTimesBlockDiag(:, 3));
plot(cumsum(blockSizes), meanTimesBlockFull(:, 3));
set(gca, 'YScale', 'log');
legend('randn()', 'block-diags', 'block-full', 'Location', 'southeast');
xlabel('Matrix Width');
ylabel('Average Compute Time');
title('blockinv, ''full'' type');

subplot(2,2,4);
hold on;
plot(cumsum(blockSizes), meanTimesRandom(:, 4));
plot(cumsum(blockSizes), meanTimesBlockDiag(:, 4));
plot(cumsum(blockSizes), meanTimesBlockFull(:, 4));
set(gca, 'YScale', 'log');
legend('randn()', 'block-diags', 'block-full', 'Location', 'southeast');
xlabel('Matrix Width');
ylabel('Average Compute Time');
title('blockinv, ''sparse'' type');

saveas(gcf, 'profileResults.fig');

%% Helpers for creating structured matrices

function M = randBlockDiags(blockSizes, isFull)
% Create matrix where each 'block' is a random diagonal matrix
blocks = cell(length(blockSizes));
for i=1:length(blockSizes)
    for j=1:length(blockSizes)
        blocks{i,j} = spdiag(randn(blockSizes(i), 1));
    end
end
M = cell2mat(blocks);
if isFull
    M = full(M);
end
end

function M = randFullBlocks(blockSizes, isFull)
% Create matrix where the block-diagonal consists of random full matrices.
blocks = cell(length(blockSizes));
for i=1:length(blockSizes)
    for j=1:length(blockSizes)
        if i == j
            blocks{i,j} = randn(blockSizes(i));
        else
            blocks{i,j} = sparse(zeros(blockSizes(i)));
        end
    end
end
M = cell2mat(blocks);
if isFull
    M = full(M);
end
end