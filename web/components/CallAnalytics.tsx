// Call Analytics Component
// Allows uploading historical calls and viewing analytics data

import { useState, useEffect } from 'react';

interface UploadedFile {
  _id: string;
  original_name: string;
  processed: boolean;
  outcome?: string;
  upload_date: string;
}

interface AnalyticsData {
  uploads: UploadedFile[];
  analytics: any[];
  summary: {
    total_uploads: number;
    processed_count: number;
    success_rate: number;
  };
}

interface CallPattern {
  type: string;
  phrase: string;
  context: string;
  outcome: string;
  source_id: string;
}

interface PatternsData {
  patterns: { [key: string]: CallPattern[] };
  total_count: number;
  types: string[];
}

export default function CallAnalytics() {
  const [activeTab, setActiveTab] = useState<'upload' | 'analytics' | 'patterns'>('upload');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [patternsData, setPatternsData] = useState<PatternsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch analytics data
  const fetchAnalyticsData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/analytics-data`);
      if (response.ok) {
        const data = await response.json();
        setAnalyticsData(data);
      }
    } catch (error) {
      console.error('Error fetching analytics data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch patterns data
  const fetchPatternsData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/call-patterns`);
      if (response.ok) {
        const data = await response.json();
        setPatternsData(data);
      }
    } catch (error) {
      console.error('Error fetching patterns data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle file upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.mp3') && !file.name.toLowerCase().endsWith('.wav')) {
      setUploadStatus('Please select an MP3 or WAV file');
      return;
    }

    setIsUploading(true);
    setUploadStatus('Uploading...');

    const formData = new FormData();
    formData.append('file', file);
    
    // Add metadata
    const metadata = {
      participants: 'rep, insurance',
      outcome: 'pending',
      date: new Date().toISOString().split('T')[0]
    };
    formData.append('metadata', JSON.stringify(metadata));

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/upload-audio`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setUploadStatus(`Upload successful! File ID: ${result.file_id}`);
        // Refresh analytics data
        if (activeTab === 'analytics') {
          fetchAnalyticsData();
        }
      } else {
        const error = await response.json();
        setUploadStatus(`Upload failed: ${error.detail}`);
      }
    } catch (error) {
      setUploadStatus('Upload failed: Network error');
    } finally {
      setIsUploading(false);
      // Clear file input
      event.target.value = '';
    }
  };

  // Load data when tab changes
  useEffect(() => {
    if (activeTab === 'analytics') {
      fetchAnalyticsData();
    } else if (activeTab === 'patterns') {
      fetchPatternsData();
    }
  }, [activeTab]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'success': return '#10b981';
      case 'failure': return '#ef4444';
      case 'pending': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{
        padding: '24px',
        borderBottom: '1px solid #e5e7eb',
        background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
        color: 'white'
      }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: '0 0 4px 0' }}>
          Call Analytics
        </h2>
        <p style={{ fontSize: '0.9rem', opacity: 0.9, margin: 0 }}>
          Upload and analyze historical prior authorization calls
        </p>
      </div>

      {/* Tab Navigation */}
      <div style={{
        padding: '0 24px',
        borderBottom: '1px solid #e5e7eb',
        background: '#f9fafb',
        display: 'flex',
        gap: '16px'
      }}>
        {(['upload', 'analytics', 'patterns'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '12px 16px',
              border: 'none',
              background: 'none',
              borderBottom: activeTab === tab ? '2px solid #8b5cf6' : '2px solid transparent',
              color: activeTab === tab ? '#8b5cf6' : '#6b7280',
              fontWeight: activeTab === tab ? '600' : '400',
              cursor: 'pointer',
              textTransform: 'capitalize',
              transition: 'all 0.2s ease'
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: '24px' }}>
        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div style={{ maxWidth: '600px' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '16px', color: '#1f2937' }}>
              Upload Historical Call Recording
            </h3>
            
            <div style={{
              border: '2px dashed #d1d5db',
              borderRadius: '12px',
              padding: '40px',
              textAlign: 'center',
              background: '#f9fafb'
            }}>
              <div style={{ marginBottom: '16px' }}>
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="1.5" style={{ margin: '0 auto' }}>
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7,10 12,15 17,10"/>
                  <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
              </div>
              
              <input
                type="file"
                accept=".mp3,.wav"
                onChange={handleFileUpload}
                disabled={isUploading}
                style={{
                  width: '100%',
                  padding: '8px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  marginBottom: '12px'
                }}
              />
              
              <p style={{ fontSize: '0.9rem', color: '#6b7280', margin: '0 0 8px 0' }}>
                Select MP3 or WAV file to upload
              </p>
              
              <p style={{ fontSize: '0.8rem', color: '#9ca3af', margin: 0 }}>
                Audio will be analyzed for patterns, success factors, and conversation flow
              </p>
            </div>

            {uploadStatus && (
              <div style={{
                marginTop: '16px',
                padding: '12px',
                borderRadius: '8px',
                background: uploadStatus.includes('successful') ? '#d1fae5' : '#fee2e2',
                color: uploadStatus.includes('successful') ? '#065f46' : '#991b1b',
                fontSize: '0.9rem'
              }}>
                {uploadStatus}
              </div>
            )}
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: '600', margin: 0, color: '#1f2937' }}>
                Call Analytics Dashboard
              </h3>
              <button
                onClick={fetchAnalyticsData}
                disabled={isLoading}
                style={{
                  padding: '8px 16px',
                  background: '#8b5cf6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.9rem'
                }}
              >
                {isLoading ? 'Loading...' : 'Refresh'}
              </button>
            </div>

            {analyticsData && (
              <>
                {/* Summary Cards */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
                  <div style={{
                    background: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    padding: '20px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#8b5cf6' }}>
                      {analyticsData.summary.total_uploads}
                    </div>
                    <div style={{ fontSize: '0.9rem', color: '#6b7280' }}>Total Uploads</div>
                  </div>
                  
                  <div style={{
                    background: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    padding: '20px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#10b981' }}>
                      {analyticsData.summary.processed_count}
                    </div>
                    <div style={{ fontSize: '0.9rem', color: '#6b7280' }}>Processed</div>
                  </div>
                  
                  <div style={{
                    background: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    padding: '20px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#f59e0b' }}>
                      {Math.round(analyticsData.summary.success_rate * 100)}%
                    </div>
                    <div style={{ fontSize: '0.9rem', color: '#6b7280' }}>Success Rate</div>
                  </div>
                </div>

                {/* Uploads Table */}
                <div style={{
                  background: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  overflow: 'hidden'
                }}>
                  <div style={{ padding: '16px', borderBottom: '1px solid #e5e7eb', background: '#f9fafb' }}>
                    <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: '600' }}>Recent Uploads</h4>
                  </div>
                  
                  {analyticsData.uploads.length > 0 ? (
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', fontSize: '0.9rem' }}>
                        <thead>
                          <tr style={{ background: '#f9fafb' }}>
                            <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #e5e7eb' }}>File Name</th>
                            <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #e5e7eb' }}>Upload Date</th>
                            <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #e5e7eb' }}>Status</th>
                            <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #e5e7eb' }}>Outcome</th>
                          </tr>
                        </thead>
                        <tbody>
                          {analyticsData.uploads.map((upload, index) => (
                            <tr key={upload._id || index}>
                              <td style={{ padding: '12px', borderBottom: '1px solid #f3f4f6' }}>
                                {upload.original_name}
                              </td>
                              <td style={{ padding: '12px', borderBottom: '1px solid #f3f4f6' }}>
                                {formatDate(upload.upload_date)}
                              </td>
                              <td style={{ padding: '12px', borderBottom: '1px solid #f3f4f6' }}>
                                <span style={{
                                  padding: '4px 8px',
                                  borderRadius: '12px',
                                  fontSize: '0.8rem',
                                  background: upload.processed ? '#d1fae5' : '#fef3c7',
                                  color: upload.processed ? '#065f46' : '#92400e'
                                }}>
                                  {upload.processed ? 'Processed' : 'Processing'}
                                </span>
                              </td>
                              <td style={{ padding: '12px', borderBottom: '1px solid #f3f4f6' }}>
                                {upload.outcome && (
                                  <span style={{
                                    padding: '4px 8px',
                                    borderRadius: '12px',
                                    fontSize: '0.8rem',
                                    background: upload.outcome === 'success' ? '#d1fae5' : '#fee2e2',
                                    color: upload.outcome === 'success' ? '#065f46' : '#991b1b'
                                  }}>
                                    {upload.outcome}
                                  </span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div style={{ padding: '40px', textAlign: 'center', color: '#6b7280' }}>
                      No uploads found. Upload some call recordings to see analytics.
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        )}

        {/* Patterns Tab */}
        {activeTab === 'patterns' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: '600', margin: 0, color: '#1f2937' }}>
                Extracted Call Patterns
              </h3>
              <button
                onClick={fetchPatternsData}
                disabled={isLoading}
                style={{
                  padding: '8px 16px',
                  background: '#8b5cf6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.9rem'
                }}
              >
                {isLoading ? 'Loading...' : 'Refresh'}
              </button>
            </div>

            {patternsData && patternsData.total_count > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {/* Summary */}
                <div style={{
                  background: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  padding: '16px'
                }}>
                  <div style={{ fontSize: '0.9rem', color: '#6b7280' }}>
                    Found <strong>{patternsData.total_count}</strong> patterns across <strong>{patternsData.types.length}</strong> categories
                  </div>
                </div>

                {/* Pattern Groups */}
                {Object.entries(patternsData.patterns).map(([type, patterns]) => (
                  <div key={type} style={{
                    background: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      padding: '16px',
                      borderBottom: '1px solid #e5e7eb',
                      background: '#f9fafb'
                    }}>
                      <h4 style={{
                        margin: 0,
                        fontSize: '1rem',
                        fontWeight: '600',
                        textTransform: 'capitalize',
                        color: '#1f2937'
                      }}>
                        {type.replace('_', ' ')} ({patterns.length})
                      </h4>
                    </div>
                    
                    <div style={{ padding: '16px' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        {patterns.slice(0, 5).map((pattern, index) => (
                          <div key={index} style={{
                            padding: '12px',
                            border: '1px solid #f3f4f6',
                            borderRadius: '6px',
                            background: '#fafafa'
                          }}>
                            <div style={{
                              fontWeight: '500',
                              color: '#1f2937',
                              marginBottom: '4px'
                            }}>
                              "{pattern.phrase}"
                            </div>
                            <div style={{
                              fontSize: '0.8rem',
                              color: '#6b7280',
                              marginBottom: '4px'
                            }}>
                              Context: {pattern.context}
                            </div>
                            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                              <span style={{
                                padding: '2px 8px',
                                borderRadius: '12px',
                                fontSize: '0.7rem',
                                background: pattern.outcome === 'success' ? '#d1fae5' : '#fee2e2',
                                color: pattern.outcome === 'success' ? '#065f46' : '#991b1b'
                              }}>
                                {pattern.outcome}
                              </span>
                            </div>
                          </div>
                        ))}
                        {patterns.length > 5 && (
                          <div style={{
                            textAlign: 'center',
                            fontSize: '0.9rem',
                            color: '#6b7280',
                            padding: '8px'
                          }}>
                            ... and {patterns.length - 5} more patterns
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{
                background: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                padding: '40px',
                textAlign: 'center',
                color: '#6b7280'
              }}>
                No patterns extracted yet. Upload and process some call recordings to see patterns.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}