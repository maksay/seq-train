% Variables that need to be specified prior to this script.
%
%SID - start frame
%FID - finish frame
%OUT - output file
%CAM - camera index
%seqmap. Should have format DukeMTMCT-debug-pid.txt. The file itself should contain path to file pid.txt in resDir.

resDir = 'res/DukeMTMCT/debug/';
gtDataDir = 'gt/DukeMTMCT';
benchmark = 'DukeMTMCT';

addpath(genpath('.'));
warning off;

% Benchmark specfific properties
world = 0;
threshold = 0.5;
multicam = 0;
if strcmp(benchmark, 'MOT15')
elseif strcmp(benchmark, 'MOT15_3D')
    world = 1;
    threshold = 1;
elseif strcmp(benchmark, 'MOT16') 
elseif strcmp(benchmark, 'MOT17')
elseif strcmp(benchmark, 'PETS2017')
elseif strcmp(benchmark, 'DukeMTMCT')
    multicam = 1;
end

% Read sequence list
sequenceListFile = fullfile('seqmaps',seqmap);
allSequences = parseSequences(sequenceListFile);
%fprintf('Sequences: \n');
%disp(allSequences')
gtMat = [];
resMat = [];

% Evaluate sequences individually
allMets = [];
for ind = 1:length(allSequences)
    
    % DukeMTMCT parsing
    ROI = getROIs();
    if strcmp(seqmap,'DukeMTMCT-test-easy.txt')
        load('gt/DukeMTMCT/testData.mat');
        gtdata = testData;
        testInterval = [263504:356648];
    elseif strcmp(seqmap,'DukeMTMCT-test-hard.txt')
        load('gt/DukeMTMCT/testHardData.mat');
        gtdata = testHardData;
        testInterval = [227541:263503];
    elseif strcmp(seqmap,'DukeMTMCT-trainval.txt')
        load('gt/DukeMTMCT/trainval.mat');
        gtdata = trainData;
        testInterval = [47720:227540]; % takes too long
    elseif strcmp(seqmap,'DukeMTMCT-trainval-mini.txt') % shorter version of trainval
        load('gt/DukeMTMCT/trainval.mat');
        gtdata = trainData;
        testInterval = [127720:187540];
    else %if strcmp(seqmap,'DukeMTMCT-debug.txt')
        load('gt/DukeMTMCT/trainval.mat');
        gtdata = trainData;
        testInterval = [SID:FID];
    end
    
    cam = CAM;
    %startTimes = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766];
    startTimes = [1, 1, 1, 1, 1, 1, 1, 1];
    filter = gtdata(:,1) == cam & ismember(gtdata(:,3) + startTimes(cam) - 1, testInterval);
    gtdata = gtdata(filter,:);
    gtdata = gtdata(:,2:end);
    gtdata(:,[1 2]) = gtdata(:,[2 1]);
    gtdata(:,1) = gtdata(:,1) + startTimes(cam) - testInterval(1); % normalize frames
    gtdata = sortrows(gtdata,[1 2]);
    gtMat{ind} = gtdata;
        
    % Duke data format
    sequenceName = allSequences{ind};
    resFilename = [resDir, sequenceName,  '.txt'];
    s = dir(resFilename); 
    if exist(resFilename,'file') && s.bytes ~= 0
        resdata = dlmread(resFilename); 
    else
        resdata = zeros(0,9);
    end
    cam = CAM;
    
    % Filter rows by frame interval
    %startTimes = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766];
    startTimes = [1, 1, 1, 1, 1, 1, 1, 1];
    resdata(~ismember(resdata(:,1) + startTimes(cam) - 1, testInterval),:) = [];
    
    % Filter rows by feet position within ROI
    feetpos = [ resdata(:,3) + 0.5*resdata(:,5), resdata(:,4) + resdata(:,6)];
    resdata = resdata(inpolygon(feetpos(:,1),feetpos(:,2), ROI{cam}(:,1),ROI{cam}(:,2)),:);
    resdata(:,1) = resdata(:,1) + startTimes(cam) - testInterval(1); % normalize frames
    resdata = sortrows(resdata,[1 2]);
    resMat{ind} = resdata;

    
    % Evaluate sequence
    [metsCLEAR, mInf, additionalInfo] = CLEAR_MOT_HUN(gtMat{ind}, resMat{ind}, threshold, world);
    metsID = IDmeasures(gtMat{ind}, resMat{ind}, threshold, world);
    mets = [metsID.IDF1, metsID.IDP, metsID.IDR, metsCLEAR];
    allMets(ind).name = sequenceName;
    allMets(ind).m    = mets;
    allMets(ind).IDmeasures = metsID;
    allMets(ind).additionalInfo = additionalInfo;
    %fprintf('%s\n', sequenceName); 
    fprintf(fopen(OUT, 'w'), '%0.3f %0.3f', metsID.IDF1, metsCLEAR(12));
    %printMetrics(mets); 
    %fprintf('\n');
    
end
