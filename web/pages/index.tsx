/**
 * index.tsx - Main dashboard for Alfons Prior Authorization Bot
 * Redesigned with shadcn/ui components and modern styling
 */

import { useState } from 'react';
import { createClient } from '@supabase/supabase-js';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { BarChart3, Phone, Users, Activity } from 'lucide-react';
import EHRInterface from '@/components/EHRInterface';
import VoiceAgentWindow from '@/components/VoiceAgentWindow';
import CallAnalytics from '@/components/CallAnalytics';

// Initialize Supabase client
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

type ViewMode = 'dashboard' | 'analytics';

export default function Home() {
  const [currentView, setCurrentView] = useState<ViewMode>('dashboard');

  if (currentView === 'analytics') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
        {/* Analytics Header */}
        <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                  <BarChart3 className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gradient">Call Analytics</h1>
                  <p className="text-sm text-muted-foreground">Historical call analysis & patterns</p>
                </div>
              </div>
              <Button 
                variant="outline" 
                onClick={() => setCurrentView('dashboard')}
                className="gap-2"
              >
                <Activity className="w-4 h-4" />
                Back to Dashboard
              </Button>
            </div>
          </div>
        </header>

        {/* Analytics Content */}
        <main className="container mx-auto px-6 py-8">
          <CallAnalytics />
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Main Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gradient-to-br from-primary to-primary/80 rounded-xl flex items-center justify-center shadow-lg">
                <Phone className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gradient">Alfons</h1>
                <p className="text-muted-foreground">AI Prior Authorization Assistant</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <Button 
                variant="outline" 
                onClick={() => setCurrentView('analytics')}
                className="gap-2 hidden sm:flex"
              >
                <BarChart3 className="w-4 h-4" />
                Analytics
              </Button>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                System Online
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="border-b bg-white/60 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                <Users className="w-4 h-4 text-green-600" />
              </div>
              <div>
                <div className="text-sm font-medium">Active Patients</div>
                <div className="text-xs text-muted-foreground">Pending authorization</div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                <Phone className="w-4 h-4 text-blue-600" />
              </div>
              <div>
                <div className="text-sm font-medium">Call Status</div>
                <div className="text-xs text-muted-foreground">Ready to assist</div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                <Activity className="w-4 h-4 text-purple-600" />
              </div>
              <div>
                <div className="text-sm font-medium">Success Rate</div>
                <div className="text-xs text-muted-foreground">AI efficiency</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Dashboard Content */}
      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 h-[calc(100vh-240px)]">
          {/* EHR Interface */}
          <Card className="flex flex-col overflow-hidden glass-effect card-hover">
            <div className="flex-1 min-h-0">
              <EHRInterface />
            </div>
          </Card>
          
          {/* Voice Agent Window */}
          <Card className="flex flex-col overflow-hidden glass-effect card-hover">
            <div className="flex-1 min-h-0">
              <VoiceAgentWindow />
            </div>
          </Card>
        </div>

        {/* Quick Actions */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="p-6 glass-effect card-hover">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg flex items-center justify-center">
                <Phone className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold">Start Authorization</h3>
                <p className="text-sm text-muted-foreground">Begin new prior auth call</p>
              </div>
            </div>
          </Card>
          
          <Card className="p-6 glass-effect card-hover">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-lg flex items-center justify-center">
                <Users className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold">Patient Queue</h3>
                <p className="text-sm text-muted-foreground">View pending requests</p>
              </div>
            </div>
          </Card>
          
          <Card className="p-6 glass-effect card-hover">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-violet-600 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold">Call Reports</h3>
                <p className="text-sm text-muted-foreground">Analytics & insights</p>
              </div>
            </div>
          </Card>
        </div>
      </main>
    </div>
  );
}