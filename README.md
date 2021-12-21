# BacteriaDetection

**Quick help: If you have** ``3Darray.mat`` **in your working folder, run the program with default parameters on the MATLAB command line by typing "BacteriaDetection".** **Otherwise, give a complex 3D array from workspace as the first parameter.**


### The MATLAB help text for the package (paraphrased):

**BacteriaDetection**     Detect and track moving bacteria from an image sequence

  There are three output files 
  - detection frames
  - mask frames
  - key points frames.

default input path: ``./3Darray.mat``
  
default output path ``./default_output/detection`` 

List of name-value parameters, default values (and possible options):

    frameType                'abs'    ('abs', 'ang')   use either magnitude or phase for imgs
    blackShiftFactor          0.2             scalar   helps with contrast between 0...1
    selectROI                true      (true, false)   hand-select region, fullscreen by default
    movingROI                true      (true, false)   region moves with the detected object
    saveVideo                true      (true, false)   save video to the set output path
    maxFPS                     30             scalar   max frames per second in video
   
    closeRect             [45, 45]        1x2 matrix   morphological closing width and height 
    openRect               20, 20]        1x2 matrix   morphological opening width and height
    minimumBlobArea            16             scalar   minimum detected blob area
    maximumBlobArea         10000             scalar   maximum detected blob area
    blobScaleFactor           1.2             scalar   (blob analysis parameter)
    blobNumLevels               8             scalar   (blob analysis parameter)
  
    minVisibleCount            10             scalar   detectable after this number of frames     
    invisibleForTooLong        30             scalar   track lost after this number of frames
    ageThreshold               30             scalar   (tracking parameter)
    visibilityThreshold       0.9             scalar   (tracking parameter)
    initialEstimateError [200, 50]        1x2 matrix   (Kalman filter parameter)
    motionNoise          [100, 25]        1x2 matrix   (Kalman filter parameter)
    measurementNoise          100             scalar   (Kalman filter parameter)   

Usage

    BacteriaDetection()                      uses default input and output file paths     
    BacteriaDetection(INPATH)                uses INPATH as the input path and the default output path
    BacteriaDetection(INPATH, OUTPATH)       uses INPATH as the input path and OUTPATH as the output path
    BacteriaDetection(Z, OUTPATH)            uses complex 3D array Z from workspace as input
    BacteriaDetection(..., 'name', 'value')  name-value pairs after 2 positional arguments

See also detectORBFteaures, imopen, imclose, vision.BlobAnalysis, configureKalmanFilter
