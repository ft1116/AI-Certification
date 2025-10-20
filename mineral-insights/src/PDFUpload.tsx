import React, { useState } from 'react';

interface PDFUploadResponse {
  message: string;
  filename: string;
  pages_processed: number;
  chunks_created: number;
}

const PDFUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<PDFUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type !== 'application/pdf') {
        setError('Please select a PDF file');
        return;
      }
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        setError('File size too large. Maximum 10MB allowed.');
        return;
      }
      setSelectedFile(file);
      setError(null);
      setUploadResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Use localhost API URL
      const API_BASE_URL = 'http://localhost:8003';
      
      const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const result: PDFUploadResponse = await response.json();
      setUploadResult(result);
      setSelectedFile(null);
      
      // Reset file input
      const fileInput = document.getElementById('pdf-upload') as HTMLInputElement;
      if (fileInput) fileInput.value = '';
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      if (file.type !== 'application/pdf') {
        setError('Please select a PDF file');
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setError('File size too large. Maximum 10MB allowed.');
        return;
      }
      setSelectedFile(file);
      setError(null);
      setUploadResult(null);
    }
  };

  return (
    <div style={{ 
      padding: '18px', 
      border: '2px dashed rgba(16, 185, 129, 0.3)', 
      borderRadius: '12px', 
      backgroundColor: 'rgba(15, 23, 42, 0.4)',
      marginBottom: '20px',
      backdropFilter: 'blur(10px)'
    }}>
      <h3 style={{ 
        color: 'white', 
        marginBottom: '6px', 
        fontSize: '16px',
        fontWeight: '700',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        📄 Upload PDF Documents
      </h3>
      
      <p style={{ 
        color: 'rgba(255, 255, 255, 0.7)', 
        fontSize: '13px', 
        marginBottom: '12px',
        margin: '0 0 12px 0'
      }}>
        Upload documents to enhance the AI's knowledge base
      </p>

      <div
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        style={{
          border: '2px dashed rgba(16, 185, 129, 0.4)',
          borderRadius: '10px',
          padding: '16px',
          textAlign: 'center',
          backgroundColor: 'rgba(30, 41, 59, 0.3)',
          marginBottom: '10px',
          cursor: 'pointer',
          transition: 'all 0.2s'
        }}
      >
        <input
          id="pdf-upload"
          type="file"
          accept=".pdf"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        
        {selectedFile ? (
          <div>
            <p style={{ color: '#10b981', fontWeight: '600', margin: '0 0 4px 0' }}>
              📎 {selectedFile.name}
            </p>
            <p style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '12px', margin: 0 }}>
              Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        ) : (
          <div>
            <p style={{ color: 'rgba(255, 255, 255, 0.8)', margin: '0 0 10px 0' }}>
              📁 Drag & drop a PDF here, or click to select
            </p>
            <button
              onClick={() => document.getElementById('pdf-upload')?.click()}
              style={{
                padding: '10px 18px',
                backgroundColor: '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '13px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
              }}
            >
              Choose File
            </button>
          </div>
        )}
      </div>

      {selectedFile && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          style={{
            width: '100%',
            padding: '12px',
            backgroundColor: uploading ? 'rgba(107, 114, 128, 0.5)' : '#10b981',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: uploading ? 'not-allowed' : 'pointer',
            fontWeight: '600',
            fontSize: '14px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
          }}
        >
          {uploading ? '⏳ Processing PDF...' : '📤 Upload & Process'}
        </button>
      )}

      {error && (
        <div style={{
          marginTop: '12px',
          padding: '12px',
          backgroundColor: 'rgba(239, 68, 68, 0.15)',
          color: '#fca5a5',
          borderRadius: '8px',
          fontSize: '13px',
          border: '1px solid rgba(239, 68, 68, 0.3)'
        }}>
          ❌ {error}
        </div>
      )}

      {uploadResult && (
        <div style={{
          marginTop: '12px',
          padding: '12px',
          backgroundColor: 'rgba(16, 185, 129, 0.15)',
          color: '#10b981',
          borderRadius: '8px',
          fontSize: '13px',
          border: '1px solid rgba(16, 185, 129, 0.3)'
        }}>
          <p style={{ margin: '0 0 6px 0', fontWeight: '600' }}>✅ {uploadResult.message}</p>
          <p style={{ margin: '2px 0', fontSize: '12px' }}>📊 Pages: {uploadResult.pages_processed}</p>
          <p style={{ margin: '2px 0', fontSize: '12px' }}>🔢 Chunks: {uploadResult.chunks_created}</p>
          <p style={{ margin: '6px 0 0 0', fontSize: '11px', opacity: 0.8 }}>
            Document added to knowledge base
          </p>
        </div>
      )}

      <div style={{
        marginTop: '12px',
        padding: '10px',
        backgroundColor: 'rgba(30, 41, 59, 0.3)',
        borderRadius: '8px',
        fontSize: '11px',
        color: 'rgba(255, 255, 255, 0.6)',
        border: '1px solid rgba(255, 255, 255, 0.1)'
      }}>
        <strong style={{ color: 'rgba(255, 255, 255, 0.8)' }}>ℹ️ Info:</strong> PDF only • 10MB limit • Vector embedding
      </div>
    </div>
  );
};

export default PDFUpload;
