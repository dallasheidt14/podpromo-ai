'use client';

import { useState, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { Youtube } from 'lucide-react';

type Props = {
  onEpisodeUploaded: (id: string) => void;
  onCompleted?: () => void;
  onClipsFetched?: (clips: any[]) => void;
  initialEpisodeId?: string;
  initialUploadStatus?: 'idle'|'uploading'|'processing'|'completed'|'error';
};

export default function EpisodeUploadSimple({
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

  // Explicit file input fallback (reliable on Safari/iOS and when dropzone click is blocked)
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onDrop = (acceptedFiles: File[]) => {
    console.log('File dropped:', acceptedFiles);
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setUploadedFile(file);
      setError(null);
      setUploadStatus('uploading');
      setUploadProgress(50);
      
      // Simulate upload process
      setTimeout(() => {
        setUploadProgress(100);
        setUploadStatus('completed');
        onCompleted?.();
        console.log('Upload completed for file:', file.name);
      }, 2000);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a', '.aac', '.flac'],
      'video/*': ['.mp4', '.mov', '.avi', '.webm']
    },
    maxSize: 500 * 1024 * 1024, // 500MB
    multiple: false
  });

  const handleYouTubeUpload = () => {
    console.log('YouTube upload clicked:', youtubeUrl);
    if (!youtubeUrl.trim()) {
      setError('Please enter a YouTube URL');
      return;
    }
    
    setError(null);
    setUploadStatus('uploading');
    setUploadProgress(50);
    
    // Simulate YouTube processing
    setTimeout(() => {
      setUploadProgress(100);
      setUploadStatus('completed');
      onCompleted?.();
      console.log('YouTube processing completed for:', youtubeUrl);
    }, 2000);
  };

  const handleModeSwitch = (mode: 'file' | 'youtube') => {
    console.log('Mode switched to:', mode);
    setUploadMode(mode);
    setError(null);
    setUploadedFile(null);
    setYoutubeUrl('');
    setUploadStatus('idle');
    setUploadProgress(0);
  };

  return (
    <div className="space-y-6">
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
            onDrop([f]);
            // reset so selecting the same file again fires onChange
            e.currentTarget.value = '';
          }
        }}
      />
      <div className="text-center">
        <h3 className="text-xl font-semibold text-gray-800 mb-2">Upload Your Episode</h3>
        <p className="text-gray-600">Choose how you&apos;d like to upload your content</p>
      </div>

      {/* Mode Selection */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={() => handleModeSwitch('file')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            uploadMode === 'file'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          Upload File
        </button>
        <button
          onClick={() => handleModeSwitch('youtube')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            uploadMode === 'youtube'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          YouTube URL
        </button>
      </div>

      {/* Upload Area */}
      {uploadMode === 'file' ? (
        <div {...getRootProps()} className="border-2 border-dashed border-blue-300 rounded-xl p-8 text-center bg-blue-50 cursor-pointer hover:bg-blue-100 transition-colors">
          <input {...getInputProps()} />
          <div className="text-blue-500 mb-4">
            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <h4 className="text-lg font-medium text-gray-800 mb-2">
            {isDragActive ? 'Drop your file here!' : 'Drop your audio/video file here or click to browse'}
          </h4>
          <p className="text-gray-600 mb-4">
            Supports MP3, WAV, M4A, MP4, MOV, AVI files up to 500MB
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
          {uploadedFile && (
            <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-600">Selected: {uploadedFile.name}</p>
            </div>
          )}
        </div>
      ) : (
        <div className="border-2 border-dashed border-blue-300 rounded-xl p-8 text-center bg-blue-50">
          <div className="text-blue-500 mb-4">
            <Youtube className="w-12 h-12 mx-auto" />
          </div>
          <h4 className="text-lg font-medium text-gray-800 mb-2">
            Enter YouTube URL
          </h4>
          <p className="text-gray-600 mb-4">
            Paste a YouTube video URL to extract audio
          </p>
          <div className="max-w-md mx-auto">
            <input
              type="url"
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <button 
              onClick={handleYouTubeUpload}
              className="mt-3 px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              Process YouTube Video
            </button>
          </div>
        </div>
      )}

      {/* Status and Progress */}
      {uploadStatus !== 'idle' && (
        <div className="bg-white rounded-lg p-4 border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">
              {uploadStatus === 'uploading' && 'Uploading...'}
              {uploadStatus === 'processing' && 'Processing...'}
              {uploadStatus === 'completed' && 'Completed!'}
              {uploadStatus === 'error' && 'Error'}
            </span>
            <span className="text-sm text-gray-500">{uploadProgress}%</span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          
          {error && (
            <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
