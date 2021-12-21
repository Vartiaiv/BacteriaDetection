% BACTERIADETECTION  Detect and track moving bacteria from an image
%                    sequence
% 
%   There are three output files 
%   - detection frames
%   - mask frames
%   - key points frames.
% 
%   default input path: './3Darray.mat' 
%   default output path './default_output/detection'
% 
%   List of name-value parameters, default values (and possible options):
%     frameType                'abs'    ('abs', 'ang')   use either magnitude or phase for imgs     
%     blackShiftFactor          0.2             scalar   helps with contrast between 0...1
%     selectROI                true      (true, false)   hand-select region, fullscreen by default
%     movingROI                true      (true, false)   region moves with the detected object
%     saveVideo                true      (true, false)   save video to the set output path
%     maxFPS                     30             scalar   max frames per second in video
%    
%     closeRect             [45, 45]        1x2 matrix   morphological closing width and height 
%     openRect               20, 20]        1x2 matrix   morphological opening width and height
%     minimumBlobArea            16             scalar   minimum detected blob area
%     maximumBlobArea         10000             scalar   maximum detected blob area
%     blobScaleFactor           1.2             scalar   (blob analysis parameter)
%     blobNumLevels               8             scalar   (blob analysis parameter)
%   
%     minVisibleCount            10             scalar   detectable after this number of frames     
%     invisibleForTooLong        30             scalar   track lost after this number of frames
%     ageThreshold               30             scalar   (tracking parameter)
%     visibilityThreshold       0.9             scalar   (tracking parameter)
%     initialEstimateError [200, 50]        1x2 matrix   (Kalman filter parameter)
%     motionNoise          [100, 25]        1x2 matrix   (Kalman filter parameter)
%     measurementNoise          100             scalar   (Kalman filter parameter)
% 
%       BACTERIADETECTION() uses default input and output file paths
%       BACTERIADETECTION(INPATH) uses INPATH as the input path and the default output path
%       BACTERIADETECTION(INPATH, OUTPATH) uses INPATH as the input path and OUTPATH as the output path
%       BACTERIADETECTION(Z, OUTPATH) uses complex 3D array Z from workspace as input
%       BACTERIADETECTION(..., 'name', 'value') name-value pairs after 2 positional arguments
% 
% See also DETECTORBFEATURES, IMOPEN, IMCLOSE, VISION.BLOBANALYSIS, CONFIGUREKALMANFILTER

function BacteriaDetection(varargin)
    %------------------------- Default Parameters ------------------------%      
    p = inputParser;  
    p.FunctionName = 'BacteriaDetection';
    % Custom parameter validation function handles
    isNumScalPos = @(x) isnumeric(x) && isscalar(x) && (x > 0);
    isNumMatPos = @(x) isnumeric(x) && ismatrix(x) ...
                                    && (sum(x(:) > 0) == numelements(x(:)));
    isValidPath = @(x) (isvector(x) && ischar(x)) || ...
                        (isscalar(x) && ischar(x));
    isNumOrPath = @(x) isValidPath(x) || isnumeric(x);
                    
    % DEFAULT Positional arguments
    z3DarrayPath  = './3Darray.mat';
    detectionPath = './default_output/detection';    
    addOptional(p, 'zInput',    z3DarrayPath,  isNumOrPath);
    addOptional(p, 'outFile',   detectionPath, isValidPath);

    % DEFAULT Parameter arguments (non-positional)
    addParameter(p, 'frameType',        'abs',  @ischar);  % abs or ang
    addParameter(p, 'blackShiftFactor', 0.2,    @isNumScalPos);
    addParameter(p, 'selectROI',        true,   @islogical);
    addParameter(p, 'movingROI',        true,   @islogical);
    addParameter(p, 'saveVideo',        true,   @islogical);
    addParameter(p, 'maxFPS',           30,     @isNumScalPos);
        
    %   Morphological operations
    addParameter(p, 'closeRect', [45, 45],  @isNumMatPos); % dilate + erode
    addParameter(p, 'openRect',  [20, 20],  @isNumMatPos); % erode + dilate  

    %   Blob analyzer parameters
    addParameter(p, 'minimumBlobArea',     16, @isNumScalPos);   % was 400
    addParameter(p, 'maximumBlobArea',  10000, @isNumScalPos);   % was 6000
    addParameter(p, 'blobScaleFactor',    1.2, @isNumScalPos);   
    addParameter(p, 'blobNumLevels',        8, @isNumScalPos);   
   
    %   Tracking parameters  
    addParameter(p, 'minVisibleCount',      10,  @isNumScalPos);  % was 4
    addParameter(p, 'invisibleForTooLong',  30,  @isNumScalPos);  % was 20
    addParameter(p, 'ageThreshold',         30,  @isNumScalPos);  % was 8
    addParameter(p, 'visibilityThreshold',  0.9, @isNumScalPos);  % was 0.6 
    
    %   Kalman filter parameters.
    %   Vectors are in form [LocationVariance, VelocityVariance] 
    addParameter(p, 'initialEstimateError', [200, 50],  @isNumMatPos);   
    addParameter(p, 'motionNoise',          [100, 25],  @isNumMatPos);   
    addParameter(p, 'measurementNoise',           100,  @isNumScalPos);   
    
    % Create all the name-value pairs to struct p.Results
    parse(p, varargin{:});
    
    % Replace optional and param values with the zInput values (and defaults if no zInput)
    zInput =                p.Results.zInput;
    outFile =               p.Results.outFile;
    
    frameType =             p.Results.frameType;
    blackShiftFactor =      p.Results.blackShiftFactor;
    selectROI =             p.Results.selectROI;
    movingROI =             p.Results.movingROI;
    saveVideo =             p.Results.saveVideo;
    maxFPS =                p.Results.maxFPS;
    
    closeRect =             p.Results.closeRect; 
    openRect =              p.Results.openRect;   
    
    minimumBlobArea =       p.Results.minimumBlobArea;
    maximumBlobArea =       p.Results.maximumBlobArea;
    blobScaleFactor =       p.Results.blobScaleFactor;
    blobNumLevels =         p.Results.blobNumLevels;
    
    minVisibleCount =       p.Results.minVisibleCount;
    invisibleForTooLong =   p.Results.invisibleForTooLong;
    ageThreshold =          p.Results.ageThreshold;
    visibilityThreshold =   p.Results.visibilityThreshold;
    
    initialEstimateError =  p.Results.initialEstimateError;
    motionNoise =           p.Results.motionNoise;
    measurementNoise =      p.Results.measurementNoise;
    
    
    %% ---------------------------- Main part ----------------------------%   
    % Load intensities z from z3DarrayPath
    z = zInput;
    if isValidPath(zInput)
        disp('loading 3D array...')
        z3Darray = load(zInput); 
        z3Darray = struct2cell(z3Darray);
        z = z3Darray{1}; % the 3D array contains complex 2D intensities
        disp('done')
    end    
    
    % store complex images into cells, magnitude and phase separately
    disp('computing frame magnitudes and phases...')
    frameStoreAbs = num2cell(abs(z), [1,2]);  % magnitude (abs values) 
    frameStoreAng = num2cell(angle(z), [1,2]);  % phase   (angles)
    nFrames = size(z, 3);
    frameIds = 1:nFrames; 
    szOut = min(size(z(:,:,1))); 
    disp('done')
    % Pre-process frames
    disp('processing frames...')
    for id = 1:length(frameStoreAbs)
        frame = frameStoreAbs{id};

        % Filter
        frame = medfilt2(frame, [5,5]);

        % Normalize and shift black level by a factor (helps contrast)
        frame = normalizeFrame(blackShiftFactor, frame);
        frameStoreAbs{id} = frame;
    end

    disp('done')
    switch frameType
        case 'abs'
            frames = frameStoreAbs;
        case 'ang'
            frames = frameStoreAng;
    end
    
    % ROI is full screen by default
    frame1 = frames{1};
    ROI = [1, 1, size(frame1)-1]; 
    % Select custom ROI to replace the default
    if selectROI
        fig = figure; im = imshow(frame1);
        h_rect = drawrectangle();
        if ~ishandle(im)
            return
        end
        ROI = h_rect.Position;
        close(fig)
    end
    
    
    %% Perform detection with the set options
    disp('initializing detection...')
    % Setup system objects for detection and video playing
    obj = setupSystemObjects();
       
    % Create an empty array of tracks.
    tracks = initializeTracks(); 
    nextId = 1; % ID of the next track
        

    % Open video object for writing
    if saveVideo                    
        open(obj.pointsWriter)
        open(obj.maskWriter)
        open(obj.frameWriter)
    end
    disp('done')
    %%
    disp('detecting')
    for id = 1:length(frames) % MAIN LOOP
        frame = frames{id};
        [centroids, bboxes, mask, points] = ...
            detectObjects(frame);  %#ok<*ASGLU>
        predictNewLocationsOfTracks()
        [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment();
   
        % Moving ROI might not be useful
        if movingROI
            updateROI();
        end
        
        updateAssignedTracks();
        updateUnassignedTracks();
        deleteLostTracks();
        createNewTracks();

        displayTrackingResults()
        
        % If any of the video windows is closed, stop the program
        if ~(isOpen(obj.framePlayer) && isOpen(obj.maskPlayer) && isOpen(obj.pointsPlayer))
            break
        end
    end % MAIN LOOP
    
    if saveVideo
        % close the video writers object after stopping
        close(obj.frameWriter)  
        close(obj.pointsWriter)
        close(obj.maskWriter) 
    end   
    
    % NESTED FUNCTIONS START HERE
    %%
    function obj = setupSystemObjects()
        % Create three video players, 
        %   one to display the video,
        %   one to display the detection mask
        %   and one to display the ORB detection points
        set(0,'showHiddenHandles','on');
        x1 = 20;  % first window x coordinate
        y1 = 300; % first window y coordinate
        ws = 400; % window size
        obj.pointsPlayer = vision.VideoPlayer('Position', [x1, y1, ws, ws]);
        obj.h_pointsPlayer = gcf;         
        ftw = obj.h_pointsPlayer.findobj('TooltipString', 'Maintain fit to window');   
        % this will search the object in the figure which has the respective 'TooltipString' parameter.
        ftw.ClickedCallback();
      
        obj.maskPlayer = vision.VideoPlayer('Position', [x1 + ws, y1, ws, ws]);
        obj.h_maskPlayer = gcf;  
        ftw = obj.h_maskPlayer.findobj('TooltipString', 'Maintain fit to window');
        ftw.ClickedCallback();

        obj.framePlayer = vision.VideoPlayer('Position', [x1 + 2*ws, y1, ws, ws]);
        obj.h_framePlayer = gcf;  
        ftw = obj.h_framePlayer.findobj('TooltipString', 'Maintain fit to window');
        ftw.ClickedCallback();  % execute the callback linked with this object

        set(0,'showHiddenHandles','off');
        
        % Create System objects for video saving
        if saveVideo
            [path, name, ext] = fileparts(outFile);
            obj.pointsWriter = VideoWriter( ...
                fullfile(path, [name, '_points.mp4']),  'MPEG-4');
            obj.maskWriter = VideoWriter( ...
                fullfile(path, [name, '_mask.mp4']),    'MPEG-4');
            obj.frameWriter = VideoWriter( ...
                fullfile(path, [name, '_frames.mp4']),  'MPEG-4');

            obj.pointsWriter.FrameRate = maxFPS;  % Frame rate
            obj.maskWriter.FrameRate = maxFPS;  % Frame rate
            obj.frameWriter.FrameRate = maxFPS;  % Frame rate

        end

        % Create System objects for blob analysis

        % Connected groups of foreground pixels are likely to correspond to
        % detected objects.  The blob analysis System object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.
        obj.blobAnalyser = vision.BlobAnalysis( ...
            'BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', minimumBlobArea, ...
            'MaximumBlobArea', maximumBlobArea);
    end


    %%
    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
    end  % initializeTracks()


    %% 
    function updateROI
        if ~isempty(tracks)
            test = tracks.bbox;
            box = double(test(1, :));
             
             [fh, fw] = size(frame);
             fac = 0.10;
             ROI = [-fac*[fw, fh] + box(1:2), 2*fac*[fw, fh] + box(3:4)];
             
             if ROI(1) + ROI(3) > fw 
                 ROI(1) = fw - ROI(3);
             end
             if ROI(2) + ROI(4) > fh
                 ROI(2) = fh - ROI(4);
             end
             if ROI(1) < 1
                 ROI(1) = 1;
             end
             if ROI(2) < 1
                 ROI(2) = 1;
             end
        end
    end
    %%
    function [centroids, bboxes, mask, points] = detectObjects(frame)               
        % Find feature key points
        features = detectORBFeatures(frame, 'ROI', ROI, ...
            'ScaleFactor', blobScaleFactor, 'NumLevels', blobNumLevels);  

        coordinates = round(features.Location);
        col = coordinates(:,1);
        row = coordinates(:,2);
        points = false(size(frame));  % logical zeroes
        ind = sub2ind(size(frame), row, col);
        points(ind) = 1;       

        % Create detection mask from key points
        % Apply morphological operations fill in holes.
        mask = imclose(points, strel('rectangle', closeRect));
        mask = imopen(mask, strel('rectangle', openRect));
        mask = imfill(mask, 'holes');

        % Perform blob analysis to find connected components.
        mask = gather(mask);  % Return back to CPU
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask); 
        
        % expand bounding boxes a bit
        expand = 30;
        bboxes(:, 3:4) = bboxes(:, 3:4) + expand;
        bboxes(:, 1:2) = bboxes(:, 1:2) - expand/2;

    end  % detectObjects(frame)


    %%
    function predictNewLocationsOfTracks()
        % Use Kalman filters to predict new locations of tracks
        
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;

            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);

            % Shift the bounding box so that its center is at
            % the predicted location.
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
        
    end  % predictNewLocationsOfTracks()

    %%
    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()

        N_tracks = length(tracks);
        N_detections = size(centroids, 1);

        % Compute the cost of assigning each detection to each track.
        cost = zeros(N_tracks, N_detections);
        for i = 1:N_tracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end

        % Solve the assignment problem using a library function.
        costOfNonAssignment = 20;  % default 20
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
        
    end  % detectionToTrackAssignment()


    %%
    function updateAssignedTracks()
        
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);

            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);

            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).bbox = bbox;

            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;

            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
        
    end  % updateAssignedTracks()


    %%
    function updateUnassignedTracks()
        
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
        
    end  % updateUnassignedTracks()


    %%
    function deleteLostTracks()
        
        if isempty(tracks)
            return;
        end

        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < visibilityThreshold) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

        % Delete lost tracks.
        tracks = tracks(~lostInds);
        
    end  % deleteLostTracks()


    %%
    function createNewTracks()
        % Here new tracks are created from unassigned detections 

%         % TEST: don't create new tracks after the first one
%         if nextId > 1
%             return
%         end
            

        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);

        for i = 1:size(centroids, 1)
           
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
                      
            % Create a Kalman filter object with ConstantVelocity model.
            % centroid is the initial location
            kalmanFilter = configureKalmanFilter( ...
                'ConstantVelocity', centroid, ...
                initialEstimateError, motionNoise, measurementNoise);

            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);

            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;                        %#ok<*AGROW>

            % Increment the next id.
            nextId = nextId + 1;
        end      
        
    end  % createNewTracks()


    %%
    function displayTrackingResults()

        % Convert the frame, mask and points to uint8 "RGB".
        normalizedFrame = double(frame)/double(max(max(frame)));
        frame = uint8(repmat(normalizedFrame .* 255, [1, 1, 3]));
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        points = uint8(repmat(points, [1, 1, 3])) .* 255;
        
        if ~isempty(tracks)

            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);

            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);

                % Get ids.
                ids = int32([reliableTracks(:).id]);

                % Create labels for objects indicating the ones for
                % which we display the predicted rather than the actual
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);

                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
                

                % Draw the objects on the mask.
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
            
        end
        frame = insertShape(frame, 'rectangle', ROI, 'color', 'r', 'LineWidth', 3);
        mask = insertShape(mask, 'rectangle', ROI, 'color', 'r', 'LineWidth', 3);
        points = insertShape(points, 'rectangle', ROI, 'color', 'r', 'LineWidth', 3);

        % Display current frame on video windows
        obj.pointsPlayer.step(points);
        obj.maskPlayer.step(mask);
        obj.framePlayer.step(frame);          
        
        % Write current frame on video to file
        if saveVideo
            writeVideo(obj.pointsWriter, points);                
            writeVideo(obj.maskWriter, mask);                
            writeVideo(obj.frameWriter, frame);                
        end  
        
        % Pause between frames
        pause_amount = 1/maxFPS;    
        if maxFPS > 0
            pause(pause_amount)
        end
        
    end % displayTrackingResults()
    
end % BacDetection

%% Utility functions

% Shift black level by factor k and normalize pixel values
% E.g. with k = 0.10 the lowest 10% pixel values are considered black
function frame = normalizeFrame(k, frame) 
    % Shift black level
    minval = min(frame, [], 'all');
    maxval = max(frame, [], 'all');
    frame = frame + abs(minval);  % make values positive, with 0 minimum
    frame = frame - k*maxval;  % shift min value to k*max

    % Normalize pixels to interval 0...1
    maxval = max(frame, [], 'all');    
    frame = double(frame / maxval);
end
