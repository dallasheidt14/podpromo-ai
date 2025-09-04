"use client";
import { useAuth } from "@/src/contexts/AuthContext";
import UsageDisplay from "./UsageDisplay";
import UpgradeButton from "./UpgradeButton";

export default function Dashboard() {
  const { user, logout } = useAuth();

  if (!user) {
    return null;
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Welcome back, {user.name}!
          </h1>
          <p className="text-gray-600 mt-1">
            Manage your podcast clips and subscription
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-500">
            {user.email}
          </span>
          <button
            onClick={logout}
            className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
          >
            Sign Out
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Usage Summary */}
        <UsageDisplay userId={user.id} />
        
        {/* Upgrade Section */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Subscription
          </h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Current Plan:</span>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                user.plan === "pro" 
                  ? "bg-green-100 text-green-800" 
                  : "bg-blue-100 text-blue-800"
              }`}>
                {user.plan === "pro" ? "Pro" : "Free"}
              </span>
            </div>
            
            {user.plan === "free" && (
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-medium text-blue-900 mb-2">
                  Upgrade to Pro
                </h4>
                <p className="text-sm text-blue-800 mb-3">
                  Get unlimited uploads and premium features
                </p>
                <UpgradeButton userId={user.id} currentPlan={user.plan} />
              </div>
            )}
            
            {user.plan === "pro" && (
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-medium text-green-900 mb-2">
                  ✓ Pro Plan Active
                </h4>
                <p className="text-sm text-green-800">
                  You have unlimited uploads and access to all features
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Quick Actions
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors text-center">
            <div className="text-2xl mb-2">📁</div>
            <div className="font-medium text-gray-900">Upload Podcast</div>
            <div className="text-sm text-gray-500">Add new episode</div>
          </button>
          
          <button className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors text-center">
            <div className="text-2xl mb-2">🎬</div>
            <div className="font-medium text-gray-900">Generate Clips</div>
            <div className="text-sm text-gray-500">AI-powered editing</div>
          </button>
          
          <button className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors text-center">
            <div className="text-2xl mb-2">📊</div>
            <div className="font-medium text-gray-900">View Analytics</div>
            <div className="text-sm text-gray-500">Track performance</div>
          </button>
        </div>
      </div>
    </div>
  );
}
