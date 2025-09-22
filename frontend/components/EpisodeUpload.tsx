'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileAudio, FileVideo, X, CheckCircle, AlertCircle, Clock, Loader2, Youtube, Link } from 'lucide-react';
import { Episode, Clip, ProgressInfo } from '@shared/types';
import { normalizeClip, normalizeProgressInfo, normalizeProgress } from '@shared/normalize';
import { getClips, uploadFile, uploadYouTube, handleApiResult, uploadYouTubeSimple, type ProgressResponse, isTerminalProgress } from '../src/shared/api';
import { createProgressPoller, PollingOptions, Poller } from '@shared/polling';
import { useProgressPoller } from '../src/hooks/useProgress';


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
  const [uploadMode, setUploadMode] = useState<'file' | 'youtube'>('file');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState<string>('');
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
  const pollerRef = useRef<Poller | null>(null);

  // Explicit file input fallback (reliable on Safari/iOS and when dropzone click is blocked)
  const fileInputRef = useRef<HTMLInputElement>(null);

  // New progress poller hook for YouTube uploads
  useProgressPoller(episodeId || undefined, {
    onUpdate: (p: ProgressResponse) => {
      const progressInfo = normalizeProgressInfo({
        stage: p.stage,
        percentage: p.percent || 0,
        message: p.message || ''
      });
      setProgressData(progressInfo);
      if (process.env.NODE_ENV === 'development') {
        console.log(`[PROGRESS_POLLER] Updated progress: ${p.stage} ${p.percent}%`);
      }
    },
    onDone: () => {
      if (process.env.NODE_ENV === 'development') {
        console.log('[PROGRESS_POLLER] Processing completed!');
      }
      setUploadStatus('completed');
      onCompleted?.();
    }
  });

  // 🔁 If parent passes an id+status, pick up polling
  useEffect(() => {
    if (initialEpisodeId) setEpisodeId(initialEpisodeId);
  }, [initialEpisodeId]);

  // Auto-fetch clips when processing completes
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      console.log('[AUTO-FETCH] Checking conditions:', {
        stage: progressData?.stage,
        percentage: progressData?.percentage,
        didFetch: didFetchRef.current,
        episodeId
      });
    }
    
    // Check for completed state with more flexible conditions
    const isCompleted = isTerminalProgress(progressData);
    
    // Handle hard errors so UI doesn't hang forever
    if (progressData?.stage === "error") {
      console.warn("[PROGRESS] Error:", progressData.message || "processing failed");
      let errorMessage = progressData.message || "Processing failed";
      
      // Add helpful context for YouTube errors
      if (errorMessage.includes("requires login") || errorMessage.includes("age restrictions")) {
        errorMessage += " Most videos work fine - try a different video or contact support if this persists.";
      }
      
      setError(errorMessage);
      setUploadStatus('error');
      return;
    }
    
    if (isCompleted && !didFetchRef.current && episodeId) {
      if (process.env.NODE_ENV === 'development') {
        console.log('[AUTO-FETCH] Progress completed, fetching clips...');
      }
      didFetchRef.current = true;
      setGeneratingClips(true);

      getClips(episodeId)
        .then(result => {
          // Handle API result format
          const rawClips = result?.ok ? result.data?.clips : (result as any)?.clips;

          // Normalize clips like the parent component does
          const normalized = Array.isArray(rawClips) ? rawClips.map(normalizeClip) : [];
          const isAd = (c: any) => Boolean(c?.is_advertisement || c?._ad_flag || c?.features?.is_advertisement);
          const ranked = normalized.filter(c => !isAd(c)).sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).slice(0, 12);

          setClips(ranked);
          setGeneratingClips(false);
          onClipsFetched?.(ranked); // Pass normalized clips to parent
          onCompleted?.();
        })
        .catch(err => {
          if (process.env.NODE_ENV === 'development') {
            console.error('[AUTO-FETCH] Failed to fetch clips:', err);
          }
          setGeneratingClips(false);
          setError('Failed to load clips');
        });
    }
  }, [progressData, episodeId, onCompleted]);

  // YouTube URL validation
  const validateYouTubeUrl = (url: string): { valid: boolean; error?: string } => {
    const trimmed = url.trim();
    if (!trimmed) return { valid: false, error: 'Please enter a YouTube URL' };
    
    const youtubeRegex = /^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\/.+/i;
    if (!youtubeRegex.test(trimmed)) {
      return { valid: false, error: 'Please enter a valid YouTube URL' };
    }
    
    return { valid: true };
  };

  // Error message mapping
  const mapYouTubeError = (error: string): string => {
    const errorMap: Record<string, string> = {
      'invalid_url': 'Please enter a valid YouTube URL',
      'too_short': 'Video is too short (minimum 60 seconds)',
      'too_long': 'Video is too long (maximum 4 hours)',
      'live_stream_not_supported': 'Live streams are not supported',
      'bot_detection': 'YouTube is blocking automated requests. Please try again in a few minutes or use a different video.',
      'video_unavailable': 'This video is not available for download.',
      'private_video': 'This video is private and cannot be processed.',
      'download_failed': 'Failed to download video. Please try again.',
      'audio_conversion_failed': 'Failed to process audio. Please try again.',
      'youtube_disabled': 'YouTube upload is currently disabled',
      'internal_error': 'An unexpected error occurred. Please try again.'
    };
    
    return errorMap[error] || error;
  };

  const onYouTubeUpload = useCallback(async (url: string) => {
    if (uploadStatus === 'uploading') return;
    
    // Validate URL first
    const validation = validateYouTubeUrl(url);
    if (!validation.valid) {
      setError(validation.error!);
      setUploadStatus('error');
      return;
    }

    // Reset state
    if (pollerRef.current) {
      pollerRef.current.stop();
      pollerRef.current = null;
    }
    
    setEpisodeId(null);
    setProgressData(null);
    setError(null);
    setUploadStatus('uploading');
    setUploadProgress(0);
    didFetchRef.current = false;

    try {
      // Upload YouTube URL using the new simple API
      const data = await uploadYouTubeSimple(url);
      
      if (process.env.NODE_ENV === 'development') {
        console.log('[YOUTUBE_UPLOAD] Upload successful, episode ID:', data.episode_id);
      }
      const episodeId = data.episode_id;

      setEpisodeId(episodeId);
      setUploadProgress(25); // Initial progress after upload
      setUploadStatus('processing');
      
      // The useProgressPoller hook will automatically start polling
      // and handle progress updates and completion
      onEpisodeUploaded(episodeId);
      
    } catch (err: any) {
      if (process.env.NODE_ENV === 'development') {
        console.error('[YOUTUBE_UPLOAD] Upload error:', err);
      }
      setError(mapYouTubeError(err.message || 'YouTube upload failed'));
      setUploadStatus('error');
    }
  }, [uploadStatus, onEpisodeUploaded]);

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
      });
      
      handleApiResult(
        result,
        (data) => {
          if (process.env.NODE_ENV === 'development') {
            console.log('[UPLOAD] Upload successful, episode ID:', data.episodeId);
          }
          const episodeId = data.episodeId;

          setEpisodeId(episodeId);
          setUploadProgress(25); // Initial progress after upload
          setUploadStatus('processing');
          
          // Start polling for progress
          const pollingOptions: PollingOptions = {
            onSuccess: (data) => {
              const progressInfo = normalizeProgressInfo(data);
              setProgressData(progressInfo);
              if (process.env.NODE_ENV === 'development') {
                console.log(`[POLLING] Updated progress: ${progressInfo?.stage} ${progressInfo?.percentage}%`);
              }
            },
            onError: (error, retryCount) => {
              if (process.env.NODE_ENV === 'development') {
                console.error(`[POLLING] Error (attempt ${retryCount}):`, error);
              }
              // Don't show error for scoring - it can take 30+ minutes
              // Only show error for actual failures, not timeouts during scoring
              if (retryCount >= 50 && !error.includes('signal timed out')) {
                setError(`Processing failed: ${error}`);
              }
            },
            on404: () => {
              if (process.env.NODE_ENV === 'development') {
                console.log('[POLLING] Episode not found, stopping');
              }
              setError('Episode not found');
              return false;
            },
            onComplete: () => {
              if (process.env.NODE_ENV === 'development') {
                console.log('[POLLING] Processing completed!');
              }
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
          if (process.env.NODE_ENV === 'development') {
            console.error('[UPLOAD] Upload failed:', error);
          }
          setError(error);
          setUploadStatus('error');
        }
      );
    } catch (err: any) {
      if (process.env.NODE_ENV === 'development') {
        console.error('[UPLOAD] Upload error:', err);
      }
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
      {/* Hidden manual file picker to bypass any dropzone click issues */}
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,video/*"
        className="sr-only"
        onChange={(e) => {
          const f = e.currentTarget.files?.[0];
          if (f) {
            // Reuse your existing onDrop pipeline
            // Note: onDrop expects File[] just like react-dropzone
            onDrop([f]);
            // reset so selecting the same file again fires onChange
            e.currentTarget.value = '';
          }
        }}
      />
      <AnimatePresence>
        {uploadStatus === 'idle' && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="rounded-2xl border border-neutral-200 bg-white shadow-card p-6"
          >
            {/* Mode Toggle */}
            <div className="flex mb-6 bg-gray-100 rounded-xl p-1">
              <button
                onClick={() => setUploadMode('file')}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  uploadMode === 'file'
                    ? 'bg-white shadow-sm text-blue-600'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Upload className="w-4 h-4" />
                Upload File
              </button>
              <button
                onClick={() => setUploadMode('youtube')}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  uploadMode === 'youtube'
                    ? 'bg-white shadow-sm text-red-600'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Youtube className="w-4 h-4" />
                YouTube Link
              </button>
            </div>

            {/* File Upload Mode */}
            {uploadMode === 'file' && (
              <div
                className="group relative rounded-xl border-2 border-dashed border-gray-300 bg-gray-50 p-8 text-center hover:border-blue-400 hover:bg-blue-50 transition-all duration-300 cursor-pointer"
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    (document.activeElement as HTMLElement)?.click();
                  }
                }}
                {...(getRootProps() as any)}
              >
                <input {...getInputProps()} />
                <div className="relative">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-100 to-purple-100 rounded-2xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300 shadow-lg">
                    <Upload className="w-8 h-8 text-blue-600" />
                  </div>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">
                  {isDragActive ? 'Drop your file here' : 'Upload Your Podcast Episode'}
                </h3>
                <p className="text-gray-600 mb-4">
                  Drag and drop your audio or video file, or click to browse
                </p>
                <div className="mt-4">
                  <button
                    type="button"
                    data-testid="file-picker-button"
                    onClick={(e) => {
                      e.stopPropagation(); // avoid dropzone root handler
                      fileInputRef.current?.click();
                    }}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg
                               focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                    disabled={uploadStatus === 'uploading'}
                  >
                    Choose file
                  </button>
                </div>
                <div className="inline-flex items-center gap-2 bg-white px-4 py-2 rounded-lg text-sm font-medium text-gray-700 border border-gray-200 mt-4">
                  <FileAudio className="w-4 h-4 text-blue-500" />
                  <span>MP3, WAV, M4A, MP4, MOV, AVI</span>
                  <span className="text-gray-400">•</span>
                  <span>Max 500MB</span>
                </div>
              </div>
            )}

            {/* YouTube Upload Mode */}
            {uploadMode === 'youtube' && (
              <div className="space-y-4">
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-red-100 to-pink-100 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                    <Youtube className="w-8 h-8 text-red-600" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">
                    Paste YouTube Link
                  </h3>
                  <p className="text-gray-600 mb-4">
                    Enter a YouTube video URL to download and process
                  </p>
                </div>
                
                <div className="space-y-3">
                  <input
                    type="url"
                    value={youtubeUrl}
                    onChange={(e) => setYoutubeUrl(e.target.value)}
                    placeholder="https://www.youtube.com/watch?v=..."
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none transition-colors"
                  />
                  
                  <button
                    onClick={() => onYouTubeUpload(youtubeUrl)}
                    disabled={!youtubeUrl.trim() || uploadStatus !== 'idle'}
                    className="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                  >
                    <Youtube className="w-4 h-4" />
                    Process YouTube Video
                  </button>
                </div>
                
                <div className="text-xs text-gray-500 text-center">
                  Supports YouTube.com and youtu.be links • Max 4 hours • Min 1 minute
                </div>
              </div>
            )}
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
                        const progressInfo = normalizeProgressInfo(data);
                        setProgressData(progressInfo);
                      },
                      onError: (error, retryCount) => {
                        if (process.env.NODE_ENV === 'development') {
                          console.error(`[POLLING] Error (attempt ${retryCount}):`, error);
                        }
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
