// Call Analytics Component
// Allows uploading historical calls and viewing analytics data

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Upload, BarChart3, Zap, FileAudio, Loader2, RefreshCw, TrendingUp, CheckCircle, Clock, XCircle } from 'lucide-react';
import { cn, formatDate } from '@/lib/utils';

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
      event.target.value = '';
    }
  };

  useEffect(() => {
    if (activeTab === 'analytics') {
      fetchAnalyticsData();
    } else if (activeTab === 'patterns') {
      fetchPatternsData();
    }
  }, [activeTab]);

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'success':
      case 'approved':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'failure':
      case 'denied':
        return <XCircle className="w-4 h-4 text-red-600" />;
      case 'pending':
        return <Clock className="w-4 h-4 text-amber-600" />;
      default:
        return <Clock className="w-4 h-4 text-gray-600" />;
    }
  };

  return (
    <div className="space-y-6">
      <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as any)} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="upload" className="gap-2">
            <Upload className="w-4 h-4" />
            Upload
          </TabsTrigger>
          <TabsTrigger value="analytics" className="gap-2">
            <BarChart3 className="w-4 h-4" />
            Analytics
          </TabsTrigger>
          <TabsTrigger value="patterns" className="gap-2">
            <Zap className="w-4 h-4" />
            Patterns
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileAudio className="w-5 h-5 text-primary" />
                Upload Historical Call Recording
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center space-y-4">
                <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto">
                  <Upload className="w-6 h-6 text-primary" />
                </div>
                
                <div className="space-y-2">
                  <h3 className="font-semibold">Upload Audio File</h3>
                  <p className="text-sm text-muted-foreground">
                    Select MP3 or WAV file to upload for analysis
                  </p>
                </div>

                <div className="space-y-4">
                  <input
                    type="file"
                    accept=".mp3,.wav"
                    onChange={handleFileUpload}
                    disabled={isUploading}
                    className="block w-full text-sm text-muted-foreground file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
                  />
                  
                  {isUploading && (
                    <div className="flex items-center gap-2 justify-center">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">Processing upload...</span>
                    </div>
                  )}
                </div>

                <p className="text-xs text-muted-foreground">
                  Audio will be analyzed for patterns, success factors, and conversation flow
                </p>
              </div>

              {uploadStatus && (
                <div className={cn(
                  "mt-4 p-3 rounded-md text-sm",
                  uploadStatus.includes('successful') 
                    ? "bg-green-50 text-green-800 border border-green-200" 
                    : "bg-red-50 text-red-800 border border-red-200"
                )}>
                  {uploadStatus}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold">Analytics Dashboard</h2>
            <Button 
              onClick={fetchAnalyticsData} 
              disabled={isLoading}
              variant="outline"
              className="gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              Refresh
            </Button>
          </div>

          {analyticsData && (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                        <FileAudio className="w-5 h-5 text-blue-600" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-blue-600">
                          {analyticsData.summary.total_uploads}
                        </p>
                        <p className="text-sm text-muted-foreground">Total Uploads</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                        <CheckCircle className="w-5 h-5 text-green-600" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-green-600">
                          {analyticsData.summary.processed_count}
                        </p>
                        <p className="text-sm text-muted-foreground">Processed</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-amber-100 rounded-lg flex items-center justify-center">
                        <TrendingUp className="w-5 h-5 text-amber-600" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-amber-600">
                          {Math.round(analyticsData.summary.success_rate * 100)}%
                        </p>
                        <p className="text-sm text-muted-foreground">Success Rate</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Uploads Table */}
              <Card>
                <CardHeader>
                  <CardTitle>Recent Uploads</CardTitle>
                </CardHeader>
                <CardContent>
                  {analyticsData.uploads.length > 0 ? (
                    <div className="space-y-3">
                      {analyticsData.uploads.map((upload, index) => (
                        <div key={upload._id || index} className="flex items-center justify-between p-4 border rounded-lg">
                          <div className="flex items-center space-x-3">
                            <FileAudio className="w-4 h-4 text-muted-foreground" />
                            <div>
                              <p className="font-medium">{upload.original_name}</p>
                              <p className="text-sm text-muted-foreground">
                                Uploaded {formatDate(upload.upload_date)}
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center space-x-3">
                            <Badge variant={upload.processed ? 'approved' : 'pending'}>
                              {upload.processed ? 'Processed' : 'Processing'}
                            </Badge>
                            {upload.outcome && (
                              <div className="flex items-center gap-1">
                                {getStatusIcon(upload.outcome)}
                                <span className="text-sm capitalize">{upload.outcome}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <FileAudio className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">
                        No uploads found. Upload some call recordings to see analytics.
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        <TabsContent value="patterns" className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold">Extracted Call Patterns</h2>
            <Button 
              onClick={fetchPatternsData} 
              disabled={isLoading}
              variant="outline"
              className="gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              Refresh
            </Button>
          </div>

          {patternsData && patternsData.total_count > 0 ? (
            <div className="space-y-6">
              {/* Summary */}
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                      <Zap className="w-5 h-5 text-purple-600" />
                    </div>
                    <div>
                      <p className="text-lg font-semibold">
                        Found <span className="text-primary">{patternsData.total_count}</span> patterns 
                        across <span className="text-primary">{patternsData.types.length}</span> categories
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Pattern Groups */}
              <div className="space-y-4">
                {Object.entries(patternsData.patterns).map(([type, patterns]) => (
                  <Card key={type}>
                    <CardHeader>
                      <CardTitle className="capitalize flex items-center gap-2">
                        <Zap className="w-4 h-4" />
                        {type.replace('_', ' ')} 
                        <Badge variant="secondary">({patterns.length})</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {patterns.slice(0, 5).map((pattern, index) => (
                          <div key={index} className="p-3 border rounded-lg bg-muted/30">
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <p className="font-medium">"{pattern.phrase}"</p>
                                <p className="text-sm text-muted-foreground mt-1">
                                  Context: {pattern.context}
                                </p>
                              </div>
                              <Badge variant={pattern.outcome === 'success' ? 'approved' : 'denied'}>
                                {pattern.outcome}
                              </Badge>
                            </div>
                          </div>
                        ))}
                        {patterns.length > 5 && (
                          <p className="text-center text-sm text-muted-foreground py-2">
                            ... and {patterns.length - 5} more patterns
                          </p>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          ) : (
            <Card>
              <CardContent className="pt-6 text-center py-12">
                <div className="w-12 h-12 bg-muted rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Zap className="w-6 h-6 text-muted-foreground" />
                </div>
                <h3 className="font-semibold mb-2">No Patterns Found</h3>
                <p className="text-muted-foreground text-sm">
                  Upload and process some call recordings to see patterns.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}