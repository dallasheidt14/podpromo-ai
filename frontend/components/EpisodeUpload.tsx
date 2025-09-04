'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileAudio, FileVideo, X, CheckCircle, AlertCircle, Clock, Loader2 } from 'lucide-react';
import { Episode, Clip, ProgressInfo } from '@shared/types';
import { normalizeClip, normalizeProgress } from '@shared/normalize';
import { fetchJsonWithBackoff } from '@shared/backoff';

type Props = {
  onEpisodeUploaded: (id: string) => void;
  onCompleted?: () => void;
  initialEpisodeId?: string;
  initialUploadStatus?: 'idle'|'uploading'|'processing'|'completed'|'error';
};

export default function EpisodeUpload({
  onEpisodeUploaded,
  onCompleted,
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

  // 🔁 If parent passes an id+status, pick up polling
  useEffect(() => {
    if (initialEpisodeId) setEpisodeId(initialEpisodeId);
  }, [initialEpisodeId]);

  useEffect(() => {
    setUploadStatus(initialUploadStatus);
  }, [initialUploadStatus]);

  // Fetch clips function with proper API endpoint and normalization
  const fetchClips = async (episodeId: string) => {
    try {
      console.log('[CLIPS] Fetching clips for episode:', episodeId);
      const clipsData = await fetchJsonWithBackoff(`/api/episodes/${episodeId}/clips`);
      console.log('[CLIPS] Retrieved clips:', clipsData);
      if (clipsData.ok && Array.isArray(clipsData.clips)) {
        const normalized = clipsData.clips.map(normalizeClip);
        
        // Filter ads and sort by score, then take top N
        const TOP_N = 12; // make this easy to change
        const isAd = (c: any) => Boolean(c?.is_advertisement || c?._ad_flag || c?.features?.is_advertisement);
        
        const ranked = [...normalized]
          .filter(c => !isAd(c))
          .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
          .slice(0, TOP_N);
        
        setClips(ranked);
        console.log(`[CLIPS] Found ${normalized.length} clips, showing top ${ranked.length} after filtering`);
        return ranked;
      }
    } catch (error) {
      console.error('[CLIPS] Failed to fetch clips:', error);
    }
    return [];
  };

  // Poll for progress updates with proper cleanup
  useEffect(() => {
    if (!episodeId || uploadStatus !== 'processing') return;

    let active = true;
    let timer: NodeJS.Timeout;
    const ctrl = new AbortController();
    let retryCount = 0;
    const maxRetries = 3;

    async function poll() {
      if (!active) return;
      
      try {
        console.log('[POLL] Starting poll for episode:', episodeId);
        const response = await fetch(`/api/progress/${episodeId}`, { signal: ctrl.signal });
        console.log('[POLL] Response status:', response.status);
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        // Use shared normalization for consistent parsing
        const data = await response.json();
        console.log('[POLL] Parsed data:', data);
        
        const progressInfo = normalizeProgress(data);
        
        if (progressInfo) {
          console.log('[POLL] Progress info:', progressInfo);
          setProgressData(progressInfo);
          setUploadProgress(progressInfo.percentage || 0);
          console.log('[POLL] Updated progress:', progressInfo.stage, progressInfo.percentage);
          
          // Check if processing is complete or failed
          if (progressInfo.stage === 'completed') {
            console.log('[POLL] Episode completed successfully');
            setUploadStatus('completed');
            
            // Persist episode ID for resume flow
            localStorage.setItem('lastEpisodeId', episodeId);
            
            // Notify parent that processing is complete
            onCompleted?.();
            return; // Stop polling
          } else if (progressInfo.stage === 'error') {
            console.log('[POLL] Episode processing failed:', progressInfo.message);
            if (active) {
              setUploadStatus('failed');
              setError(progressInfo.message);
            }
            return; // Stop polling
          }
        } else {
          console.warn('[POLL] No progress info found in response:', data);
        }
        
        retryCount = 0; // Reset on success
        
        // Schedule next poll with adaptive interval
        if (active) {
          const interval = getPollingInterval(progressInfo?.stage || 'processing');
          timer = setTimeout(poll, interval);
        }
        
      } catch (err) {
        if (!active) return;
        
        console.error('[POLL] Failed to fetch progress:', err);
        retryCount++;
        
        if (retryCount >= maxRetries) {
          console.error('[POLL] Max retries reached, stopping polling');
          setError('Lost connection to server. Please refresh.');
          setUploadStatus('failed');
        } else {
          console.log(`[POLL] Retry ${retryCount}/${maxRetries} after error:`, err);
          if (active) {
            timer = setTimeout(poll, 2000 * retryCount); // Exponential backoff
          }
        }
      }
    }

    poll();

    return () => {
      active = false;
      if (timer) clearTimeout(timer);
      ctrl.abort(); // cancel in-flight fetch
    };
  }, [episodeId, uploadStatus]);

  // Ensure fetchClips is called exactly once when status hits completed
  useEffect(() => {
    if (!episodeId || uploadStatus !== 'processing') return;
    const stage = progressData?.stage?.toLowerCase();
    if (stage === 'completed' && !didFetchRef.current) {
      didFetchRef.current = true;
      fetchClips(episodeId);
    }
  }, [episodeId, uploadStatus, progressData]);

  // Adaptive polling intervals
  const getPollingInterval = (stage: string): number => {
    switch (stage) {
      case 'uploading': return 500;
      case 'transcribing':
      case 'transcription': return 3000;
      case 'processing':
      case 'scoring': return 2000;
      case 'generating':
      case 'finalizing': return 1000;
      case 'completed': return 5000;
      default: return 1500;
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

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

      // Upload file
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = 'Upload failed';
        try {
          const errorText = await response.text();
          // Try to parse as JSON first
          try {
            const errorData = JSON.parse(errorText);
            errorMessage = errorData.detail || errorData.message || 'Upload failed';
          } catch {
            // If not JSON, use the raw text
            errorMessage = `HTTP ${response.status}: ${errorText.slice(0, 200)}`;
          }
        } catch (e) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const uploadResponse = await response.json();
      console.log('[UPLOAD] Full upload response:', uploadResponse);
      
      // Try multiple possible field names for episode ID
      const episodeId = uploadResponse.episodeId || 
                        uploadResponse.episode_id || 
                        uploadResponse.id ||
                        uploadResponse.data?.episodeId ||
                        uploadResponse.data?.id;
      
      console.log('[UPLOAD] Extracted episode ID:', episodeId);
      
      if (!episodeId) {
        console.error('[UPLOAD] No episode ID in response:', uploadResponse);
        throw new Error('No episode ID returned from server');
      }
      
      setEpisodeId(episodeId);
      console.log('[UPLOAD] Episode ID set:', episodeId);
      setUploadProgress(25); // Initial progress after upload
      setUploadStatus('processing');

      // Notify parent of new episode
      onEpisodeUploaded(episodeId);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploadStatus('failed');
      setUploadProgress(0);
    }
  }, [onEpisodeUploaded]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a'],
      'video/*': ['.mp4', '.mov', '.avi']
    },
    multiple: false,
    disabled: uploadStatus === 'uploading' || uploadStatus === 'processing'
  });

  const isValidFile = (file: File): boolean => {
    const validTypes = [
      'audio/mp3', 'audio/wav', 'audio/m4a', 'audio/mpeg',
      'video/mp4', 'video/mov', 'video/avi', 'video/quicktime'
    ];
    const validExtensions = ['.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi'];
    
    return validTypes.includes(file.type) || 
           validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
  };

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('audio/')) {
      return <FileAudio className="w-8 h-8 text-primary-600" />;
    }
    if (file.type.startsWith('video/')) {
      return <FileVideo className="w-8 h-8 text-secondary-600" />;
    }
    return <FileAudio className="w-8 h-8 text-gray-600" />;
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };



  const resetUpload = () => {
    setUploadedFile(null);
    setUploadStatus('idle');
    setUploadProgress(0);
    setError(null);
    setEpisode(null);
    setEpisodeId(null);
    setProgressData(null);
    setClips([]);
    setGeneratingClips(false);
    clipsFetchedRef.current = false; // Reset duplicate guard
  };

  const getProgressColor = (stage: string) => {
    switch (stage) {
      case 'uploading': return 'bg-blue-500';
      case 'converting': return 'bg-yellow-500';
      case 'transcribing': return 'bg-purple-500';
      case 'processing': return 'bg-indigo-500';
      case 'completed': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getProgressIcon = (stage: string) => {
    switch (stage) {
      case 'uploading': return <Upload className="w-6 h-6 text-white/80" />;
      case 'converting': return <Loader2 className="w-6 h-6 text-white/80 animate-spin" />;
      case 'transcribing': return <Loader2 className="w-6 h-6 text-white/80 animate-spin" />;
      case 'processing': return <Loader2 className="w-6 h-6 text-white/80 animate-spin" />;
      case 'completed': return <CheckCircle className="w-6 h-6 text-green-400" />;
      default: return <Clock className="w-6 h-6 text-white/60" />;
    }
  };

  return (
    <div className="rounded-2xl border border-[#1e2636] bg-white/[0.04] shadow-[0_10px_30px_rgba(0,0,0,0.35)] p-6 text-white">
      <div className="text-center mb-6">
        <h3 className="text-xl font-semibold text-white">Upload Your Episode</h3>
        <p className="text-sm text-white/70 mt-1">
          Drag and drop your audio or video file, or click to browse
        </p>
      </div>

      <AnimatePresence mode="wait">
        {uploadStatus === 'idle' && (
          <motion.div
            key="upload-zone"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <label
              htmlFor="file"
              {...getRootProps()}
              className="mt-4 block rounded-xl border-2 border-dashed border-white/15 bg-white/[0.03] p-6 text-center transition 
                         hover:border-white/40 hover:bg-white/[0.06] cursor-pointer"
            >
              <input {...getInputProps()} />
              <Upload className="w-12 h-12 text-white/40 mx-auto mb-4" />
              <p className="text-white/90">
                {isDragActive ? 'Drop your file here' : 'Choose a file or drag it here'}
              </p>
              <p className="text-xs text-white/55 mt-1">
                Supports MP3, WAV, M4A, MP4, MOV, AVI (max 500MB)
              </p>
            </label>
          </motion.div>
        )}

        {uploadStatus === 'uploading' && (
          <motion.div
            key="uploading"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="text-center"
          >
            <div className="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center mx-auto mb-4">
              <Upload className="w-8 h-8 text-white animate-bounce" />
            </div>
            <h4 className="text-lg font-medium text-white mb-2">Uploading...</h4>
            <div className="mt-4 rounded-lg bg-white/[0.06]">
              <div className="h-2 w-full rounded-b-lg bg-white/10 overflow-hidden">
                <div className="h-2 bg-white/80 transition-[width] duration-500" style={{ width: `${uploadProgress}%` }}/>
              </div>
            </div>
            <p className="mt-2 text-xs text-white/70">{uploadProgress}% complete</p>
          </motion.div>
        )}

        {uploadStatus === 'processing' && (
          <motion.div
            key="processing"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="text-center"
          >
            <div className="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center mx-auto mb-4">
              {getProgressIcon(progressData?.stage || 'processing')}
            </div>
            <h4 className="text-lg font-medium text-white mb-2">
              {progressData?.stage ? progressData.stage.charAt(0).toUpperCase() + progressData.stage.slice(1) : 'Processing Episode'}
            </h4>
            <p className="muted mb-4">
              {progressData?.message || 'Transcribing audio and analyzing content...'}
            </p>
            
            {/* Real-time progress bar */}
            <div className="mt-4 rounded-lg bg-white/[0.06]">
              <div className="h-2 w-full rounded-b-lg bg-white/10 overflow-hidden">
                <div className="h-2 bg-white/80 transition-[width] duration-500" style={{ width: `${uploadProgress}%` }}/>
              </div>
            </div>
            
            <div className="flex items-center justify-center space-x-2 mb-2">
              <span className="text-lg font-semibold text-white">{uploadProgress.toFixed(1)}%</span>
              <span className="muted">complete</span>
            </div>
            
            {/* Stage indicator */}
            {progressData && (
              <div className="card-2 rounded-lg p-3 text-sm muted">
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${getProgressColor(progressData.stage)}`}></div>
                  <span className="capitalize text-white/80">{progressData.stage}</span>
                </div>
                <p className="mt-1 text-xs muted">{progressData.message}</p>
              </div>
            )}
          </motion.div>
        )}

        {uploadStatus === 'completed' && uploadedFile && (
          <motion.div
            key="completed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="text-center"
          >
            <div className="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="w-8 h-8 text-green-400" />
            </div>
            <h4 className="text-lg font-medium text-white mb-2">Upload Complete!</h4>
            <div className="card-2 rounded-lg p-4 mb-4">
              <div className="flex items-center space-x-3">
                {getFileIcon(uploadedFile)}
                <div className="text-left">
                  <p className="font-medium text-white">{uploadedFile.name}</p>
                  <p className="muted">{formatFileSize(uploadedFile.size)}</p>
                </div>
              </div>
            </div>
            <div className="flex justify-center">
              <button
                onClick={resetUpload}
                className="bg-green-500 hover:bg-green-600 text-white rounded-lg px-4 py-2 transition-colors"
              >
                Upload Another File
              </button>
            </div>
          </motion.div>
        )}



        {uploadStatus === 'failed' && (
          <motion.div
            key="failed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="text-center"
          >
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <AlertCircle className="w-8 h-8 text-red-600" />
            </div>
            <h4 className="text-lg font-medium text-white mb-2">Upload Failed</h4>
            <p className="text-sm text-red-400 mb-4">{error}</p>
            <button
              onClick={resetUpload}
              className="bg-green-500 hover:bg-green-600 text-white rounded-lg px-4 py-2 transition-colors"
            >
              Try Again
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
