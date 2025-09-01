"use client";
import { useEffect, useState } from "react";

interface UsageSummary {
  user_id: string;
  plan: string;
  current_month: {
    uploads: number;
    clips_generated: number;
  };
  limits: {
    uploads_per_month: number;
    description: string;
  };
  can_upload: boolean;
  remaining_uploads?: number;
}

interface UsageDisplayProps {
  userId: string;
}

export default function UsageDisplay({ userId }: UsageDisplayProps) {
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUsage = async () => {
      try {
        const response = await fetch(`/api/usage/summary/${userId}`);
        if (response.ok) {
          const data = await response.json();
          setUsage(data);
        }
      } catch (error) {
        console.error("Failed to fetch usage:", error);
      } finally {
        setLoading(false);
      }
    };

    if (userId) {
      fetchUsage();
    }
  }, [userId]);

  if (loading) {
    return (
      <div className="bg-gray-100 p-4 rounded-lg">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-300 rounded w-1/4 mb-2"></div>
          <div className="h-3 bg-gray-300 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  if (!usage) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2 rounded-lg">
        Failed to load usage information
      </div>
    );
  }

  const isPro = usage.plan === "pro";
  const uploadsUsed = usage.current_month.uploads;
  const uploadsLimit = usage.limits.uploads_per_month;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Usage Summary
      </h3>
      
      <div className="space-y-4">
        {/* Plan Status */}
        <div className="flex items-center justify-between">
          <span className="text-gray-600">Current Plan:</span>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
            isPro 
              ? "bg-green-100 text-green-800" 
              : "bg-blue-100 text-blue-800"
          }`}>
            {isPro ? "Pro" : "Free"}
          </span>
        </div>

        {/* Uploads */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-gray-600">Uploads this month:</span>
            <span className="font-medium">
              {uploadsUsed} / {isPro ? "∞" : uploadsLimit}
            </span>
          </div>
          
          {!isPro && (
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${Math.min((uploadsUsed / uploadsLimit) * 100, 100)}%` }}
              ></div>
            </div>
          )}
          
          {!isPro && usage.remaining_uploads !== undefined && (
            <p className="text-sm text-gray-500">
              {usage.remaining_uploads} uploads remaining this month
            </p>
          )}
        </div>

        {/* Clips Generated */}
        <div className="flex items-center justify-between">
          <span className="text-gray-600">Clips generated:</span>
          <span className="font-medium">{usage.current_month.clips_generated}</span>
        </div>

        {/* Upload Status */}
        <div className="pt-2">
          {usage.can_upload ? (
            <div className="text-green-600 text-sm">
              ✓ You can upload files
            </div>
          ) : (
            <div className="text-red-600 text-sm">
              ✗ Upload limit reached for this month
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
