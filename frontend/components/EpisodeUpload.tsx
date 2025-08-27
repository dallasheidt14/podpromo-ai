'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileAudio, FileVideo, X, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import { Episode } from '../../shared/types';

interface EpisodeUploadProps {
  onEpisodeUploaded: (episode: Episode) => void;
}

export default function EpisodeUpload({ onEpisodeUploaded }: EpisodeUploadProps) {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'completed' | 'failed'>('idle');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [episode, setEpisode] = useState<Episode | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploadedFile(file);
    setError(null);
    setUploadStatus('uploading');
    setUploadProgress(0);

    try {
      // Validate file
      if (!isValidFile(file)) {
        throw new Error('Invalid file type. Please upload an audio or video file.');
      }

      if (file.size > 500 * 1024 * 1024) { // 500MB limit
        throw new Error('File too large. Maximum size is 500MB.');
      }

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Upload file
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const uploadResponse = await response.json();
      setUploadProgress(100);
      setUploadStatus('processing');

      // Create episode object
      const newEpisode: Episode = {
        id: uploadResponse.episodeId,
        filename: uploadResponse.filename,
        originalName: file.name,
        size: file.size,
        duration: undefined,
        status: 'processing',
        uploadedAt: new Date().toISOString(),
      };

      setEpisode(newEpisode);

      // Simulate processing time
      setTimeout(() => {
        setUploadStatus('completed');
        newEpisode.status = 'completed';
        newEpisode.duration = Math.random() * 300 + 180; // Random duration between 3-8 minutes
        onEpisodeUploaded(newEpisode);
      }, 3000);

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
  };

  return (
    <div className="card">
      <div className="text-center mb-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-2">Upload Your Episode</h3>
        <p className="text-gray-600">
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
            <div
              {...getRootProps()}
              className={`upload-zone ${isDragActive ? 'active' : ''}`}
            >
              <input {...getInputProps()} />
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-lg font-medium text-gray-900 mb-2">
                {isDragActive ? 'Drop your file here' : 'Choose a file or drag it here'}
              </p>
              <p className="text-sm text-gray-500">
                Supports MP3, WAV, M4A, MP4, MOV, AVI (max 500MB)
              </p>
            </div>
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
            <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Upload className="w-8 h-8 text-primary-600 animate-bounce" />
            </div>
            <h4 className="text-lg font-medium text-gray-900 mb-2">Uploading...</h4>
            <div className="progress-bar mb-4">
              <div 
                className="progress-fill" 
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-600">{uploadProgress}% complete</p>
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
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Clock className="w-8 h-8 text-blue-600 animate-spin" />
            </div>
            <h4 className="text-lg font-medium text-gray-900 mb-2">Processing Episode</h4>
            <p className="text-sm text-gray-600 mb-4">
              Transcribing audio and analyzing content...
            </p>
            <div className="progress-bar">
              <div className="progress-fill w-full animate-pulse"></div>
            </div>
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
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="w-8 h-8 text-green-600" />
            </div>
            <h4 className="text-lg font-medium text-gray-900 mb-2">Upload Complete!</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <div className="flex items-center space-x-3">
                {getFileIcon(uploadedFile)}
                <div className="text-left">
                  <p className="font-medium text-gray-900">{uploadedFile.name}</p>
                  <p className="text-sm text-gray-600">{formatFileSize(uploadedFile.size)}</p>
                </div>
              </div>
            </div>
            <button
              onClick={resetUpload}
              className="btn-outline"
            >
              Upload Another File
            </button>
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
            <h4 className="text-lg font-medium text-gray-900 mb-2">Upload Failed</h4>
            <p className="text-sm text-red-600 mb-4">{error}</p>
            <button
              onClick={resetUpload}
              className="btn-primary"
            >
              Try Again
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
