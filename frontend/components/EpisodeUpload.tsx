'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileAudio, FileVideo, X, CheckCircle, AlertCircle, Clock, Loader2 } from 'lucide-react';
import { Episode, Clip, ProgressInfo } from '../src/shared/types';
import { normalizeClip, normalizeProgressInfo, normalizeProgress } from '../src/shared/normalize';
import { getClips, uploadFile, handleApiResult } from '../src/shared/api';
import { createProgressPoller, PollingOptions } from '../src/shared/polling';

type Props = {
  onEpisodeUploaded: (id: string) => void;
  onCompleted?: () => void;
  onClipsFetched?: (clips: Clip[]) => void;
  initialEpisodeId?: string;
  initialUploadStatus?: 'idle'|'uploading'|'processing'|'completed'|'error';
};

export default function EpisodeUpload({
  onEpisodeUploaded,
  onCompleted,
  onClipsFetched,
  initialEpisodeId,
  initialUploadStatus = 'idle',
}: Props) {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState(initialUploadStatus);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [episodeId, setEpisodeId] = useState<string | null>(initialEpisodeId ?? null);
  const [progressData, setProgressData] = useState<ProgressInfo | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [generatingClips, setGeneratingClips] = useState(false);
  
  // Guard against duplicate clips fetch
  const clipsFetchedRef = useRef(false);
  const didFetchRef = useRef(false);
  const pollerRef = useRef<any>(null);

  // 🔁 If parent passes an id+status, pick up polling
  useEffect(() => {
    if (initialEpisodeId) setEpisodeId(initialEpisodeId);
  }, [initialEpisodeId]);

  // Auto-fetch clips when processing completes
  useEffect(() => {
    console.log('[AUTO-FETCH] Checking conditions:', {
      stage: progressData?.stage,
      percentage: progressData?.percentage,
      didFetch: didFetchRef.current,
      episodeId
    });
    
    // Check for completed state with more flexible conditions
    const isCompleted = progressData?.stage === 'completed' || 
                       (progressData?.percentage && progressData.percentage >= 100 && progressData?.stage !== 'error');
    
    if (isCompleted && !didFetchRef.current && episodeId) {
      console.log('[AUTO-FETCH] Progress completed, fetching clips...');
      didFetchRef.current = true;
      setGeneratingClips(true);
      
      getClips(episodeId)
        .then(result => {
          console.log('[AUTO-FETCH] Clips fetched:', result);
          // Handle API result format
          const rawClips = result?.ok ? result.data?.clips : (result as any)?.clips;
          console.log('[AUTO-FETCH] Raw clips:', rawClips);
          console.log('[AUTO-FETCH] Raw clips length:', rawClips?.length);
          console.log('[AUTO-FETCH] First raw clip structure:', rawClips?.[0]);
          
          // Normalize clips like the parent component does
          const normalized = Array.isArray(rawClips) ? rawClips.map(normalizeClip) : [];
          const isAd = (c: any) => Boolean(c?.is_advertisement || c?._ad_flag || c?.features?.is_advertisement);
          const ranked = normalized.filter(c => !isAd(c)).sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).slice(0, 12);
          
          console.log('[AUTO-FETCH] Normalized clips:', ranked);
          console.log('[AUTO-FETCH] Normalized clips length:', ranked?.length);
          console.log('[AUTO-FETCH] First normalized clip:', ranked?.[0]);
          
          setClips(ranked);
          setGeneratingClips(false);
          onClipsFetched?.(ranked); // Pass normalized clips to parent
          onCompleted?.();
        })
        .catch(err => {
          console.error('[AUTO-FETCH] Failed to fetch clips:', err);
          setGeneratingClips(false);
          setError('Failed to load clips');
        });
    }
  }, [progressData, episodeId, onCompleted]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (uploadStatus === 'uploading') return;
    if (acceptedFiles.length === 0) return;

    // Reset state
    if (pollerRef.current) {
      pollerRef.current.stop();
      pollerRef.current = null;
    }
    
    setEpisodeId(null);
    setProgressData(null);
    setError(null);
    setUploadStatus('idle');
    setUploadProgress(0);
    didFetchRef.current = false;

    const file = acceptedFiles[0];
    setUploadedFile(file);
    setError(null);
    setUploadStatus('uploading');
    setUploadProgress(0);
    setProgressData(null);

    try {
      // Validate file
      if (!isValidFile(file)) {
        throw new Error('Invalid file type. Please upload an audio or video file.');
      }

      if (file.size > 500 * 1024 * 1024) { // 500MB limit
        throw new Error('File too large. Maximum size is 500MB.');
      }

      // Upload file using API adapter with progress tracking
      const result = await uploadFile(file, (progress) => {
        setUploadProgress(progress);
        console.log(`[UPLOAD] Progress: ${progress}%`);
      });
      
      handleApiResult(
        result,
        (data) => {
          console.log('[UPLOAD] Upload successful, episode ID:', data.episodeId);
          const episodeId = data.episodeId;
          
          setEpisodeId(episodeId);
          console.log('[UPLOAD] Episode ID set:', episodeId);
          setUploadProgress(25); // Initial progress after upload
          setUploadStatus('processing');
          
          // Start polling for progress
          const pollingOptions: PollingOptions = {
            onSuccess: (data) => {
              console.log('[POLLING] Success response:', data);
              console.log('[POLLING] Raw progress data:', data?.progress);
              const progressInfo = normalizeProgressInfo(data);
              console.log('[POLLING] Progress info:', progressInfo);
              setProgressData(progressInfo);
              console.log(`[POLLING] Updated progress: ${progressInfo?.stage} ${progressInfo?.percentage}%`);
            },
            onError: (error, retryCount) => {
              console.error(`[POLLING] Error (attempt ${retryCount}):`, error);
              // Don't show error for scoring - it can take 30+ minutes
              // Only show error for actual failures, not timeouts during scoring
              if (retryCount >= 50 && !error.includes('signal timed out')) {
                setError(`Processing failed: ${error}`);
              }
            },
            on404: () => {
              console.log('[POLLING] Episode not found, stopping');
              setError('Episode not found');
              return false;
            },
            onComplete: () => {
              console.log('[POLLING] Processing completed!');
              setUploadStatus('completed');
              onCompleted?.();
            }
          };
          
          const poller = createProgressPoller(`http://localhost:8000/api/progress/${episodeId}`, pollingOptions);
          pollerRef.current = poller;
          poller.start();
          
          onEpisodeUploaded(episodeId);
        },
        (error) => {
          console.error('[UPLOAD] Upload failed:', error);
          setError(error);
          setUploadStatus('error');
        }
      );
    } catch (err: any) {
      console.error('[UPLOAD] Upload error:', err);
      setError(err.message || 'Upload failed');
      setUploadStatus('error');
    }
  }, [uploadStatus, onEpisodeUploaded]);

  const isValidFile = (file: File): boolean => {
    const validTypes = [
      'audio/mp3', 'audio/wav', 'audio/m4a', 'audio/mpeg',
      'video/mp4', 'video/mov', 'video/avi', 'video/quicktime'
    ];
    const validExtensions = ['.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi'];
    
    return validTypes.includes(file.type) || 
           validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a'],
      'video/*': ['.mp4', '.mov', '.avi']
    },
    multiple: false,
    disabled: uploadStatus === 'uploading'
  });

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollerRef.current) {
        pollerRef.current.stop();
      }
    };
  }, []);

  return (
    <div className="w-full max-w-2xl mx-auto">
      <AnimatePresence>
        {uploadStatus === 'idle' && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="group relative rounded-2xl border border-neutral-200 bg-white shadow-card p-6 text-center hover:border-blue-400 hover:bg-gradient-to-br hover:from-blue-50 hover:to-purple-50 transition-all duration-500 cursor-pointer hover:shadow-card-hover"
            {...(getRootProps() as any)}
          >
            <input {...getInputProps()} />
            <div className="relative">
              <div className="w-24 h-24 bg-gradient-to-br from-blue-100 to-purple-100 rounded-3xl flex items-center justify-center mx-auto mb-8 group-hover:scale-110 transition-transform duration-500 shadow-lg">
                <Upload className="w-12 h-12 text-blue-600" />
              </div>
              <div className="absolute -top-2 -right-2 w-8 h-8 bg-gradient-to-r from-green-500 to-blue-500 rounded-full flex items-center justify-center shadow-lg">
                <div className="w-3 h-3 bg-white rounded-full"></div>
              </div>
            </div>
            <h3 className="text-3xl font-bold text-gray-900 mb-4">
              {isDragActive ? 'Drop your file here' : 'Upload Your Podcast Episode'}
            </h3>
            <p className="text-lg text-gray-600 mb-8 max-w-lg mx-auto leading-relaxed">
              Drag and drop your audio or video file, or click to browse
            </p>
            <div className="inline-flex items-center gap-3 bg-gradient-to-r from-gray-50 to-gray-100 px-6 py-3 rounded-2xl text-sm font-medium text-gray-700 border border-gray-200">
              <FileAudio className="w-5 h-5 text-blue-500" />
              <span>MP3, WAV, M4A, MP4, MOV, AVI</span>
              <span className="text-gray-400">•</span>
              <span>Max 500MB</span>
            </div>
          </motion.div>
        )}

        {uploadStatus === 'uploading' && (
          <motion.div
            key="uploading"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="rounded-2xl border border-neutral-200 bg-white shadow-card p-6 text-center"
          >
            <div className="w-24 h-24 bg-gradient-to-br from-blue-100 to-purple-100 rounded-3xl flex items-center justify-center mx-auto mb-8 shadow-lg">
              <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
            </div>
            <h3 className="text-3xl font-bold text-gray-900 mb-4">Uploading Your Episode</h3>
            <p className="text-lg text-gray-600 mb-8">Please wait while we upload your file...</p>
            <div className="w-full max-w-md mx-auto">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Progress</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-neutral-200">
                <div 
                  className="h-2 rounded-full bg-primary-500 transition-all" 
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          </motion.div>
        )}

        {uploadStatus === 'processing' && !error && !generatingClips && (
          <motion.div
            key="processing"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="rounded-2xl border border-neutral-200 bg-white shadow-card p-6 text-center"
          >
            <div className="w-24 h-24 bg-gradient-to-br from-blue-100 to-purple-100 rounded-3xl flex items-center justify-center mx-auto mb-8 shadow-lg">
              <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
            </div>
            <h3 className="text-3xl font-bold text-gray-900 mb-4">
              {progressData?.stage === 'converting' && 'Converting Your File'}
              {progressData?.stage === 'transcribing' && 'Transcribing Audio'}
              {progressData?.stage === 'scoring' && 'Finding Viral Moments'}
              {progressData?.stage === 'processing' && 'Processing Episode'}
              {!progressData?.stage && 'Processing Your Episode'}
            </h3>
            <p className="text-lg text-gray-600 mb-8">
              {progressData?.stage === 'converting' && 'Converting your file to the optimal format...'}
              {progressData?.stage === 'transcribing' && 'Converting speech to text with AI...'}
              {progressData?.stage === 'scoring' && 'Analyzing content for viral potential... This can take 10-30 minutes for longer episodes.'}
              {progressData?.stage === 'processing' && 'Processing your episode...'}
              {!progressData?.stage && 'This may take 10-30 minutes for longer files. We\'ll keep checking in the background.'}
            </p>
            <div className="w-full max-w-md mx-auto">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Progress</span>
                <span>{progressData?.percentage || 0}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-neutral-200">
                <div 
                  className="h-2 rounded-full bg-primary-500 transition-all" 
                  style={{ width: `${progressData?.percentage || 0}%` }}
                />
              </div>
              {progressData?.message && (
                <p className="text-xs text-gray-500 mt-3">{progressData.message}</p>
              )}
              
            </div>
          </motion.div>
        )}

        {uploadStatus === 'completed' && clips.length > 0 && (
          <motion.div
            key="completed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="rounded-2xl border border-neutral-200 bg-white shadow-card p-6 text-center"
          >
            <CheckCircle className="w-8 h-8 text-green-400 mx-auto mb-4" />
            <p className="text-gray-900 mb-2">Processing complete!</p>
            <p className="text-sm text-gray-600">Found {clips.length} viral clips</p>
          </motion.div>
        )}

        {error && (
          <motion.div
            key="error"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="rounded-2xl border border-neutral-200 bg-white shadow-card p-6 text-center"
          >
            <AlertCircle className="w-8 h-8 text-red-400 mx-auto mb-4" />
            <p className="text-gray-900 mb-2">Something went wrong</p>
            <p className="text-sm text-gray-600">{error}</p>
            <div className="flex gap-2 justify-center">
              <button
                onClick={() => {
                  setError(null);
                  // Continue polling if we have an episodeId
                  if (episodeId) {
                    const pollingOptions: PollingOptions = {
                      onSuccess: (data) => {
                        console.log('[POLLING] Success response:', data);
                        const progressInfo = normalizeProgressInfo(data);
                        setProgressData(progressInfo);
                      },
                      onError: (error, retryCount) => {
                        console.error(`[POLLING] Error (attempt ${retryCount}):`, error);
                        // Don't show error for scoring timeouts
                        if (retryCount >= 50 && !error.includes('signal timed out')) {
                          setError(`Processing failed: ${error}`);
                        }
                      },
                      on404: () => {
                        setError('Episode not found');
                        return false;
                      }
                    };
                    
                    const poller = createProgressPoller(`http://localhost:8000/api/progress/${episodeId}`, pollingOptions);
                    pollerRef.current = poller;
                    poller.start();
                  }
                }}
                className="mt-4 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors shadow-sm"
              >
                Continue Processing
              </button>
              <button
                onClick={() => {
                  setError(null);
                  setUploadStatus('idle');
                  setProgressData(null);
                  setClips([]);
                  didFetchRef.current = false;
                }}
                className="mt-4 px-4 py-2 bg-white border border-gray-300 hover:bg-gray-50 rounded-lg text-gray-700 transition-colors shadow-sm"
              >
                Start Over
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
