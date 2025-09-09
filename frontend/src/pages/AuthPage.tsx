"use client";
import { useState } from "react";
import dynamic from "next/dynamic";
import LoginForm from "@/src/components/LoginForm";
import SignupForm from "@/src/components/SignupForm";

// Dynamically import components that use context to avoid SSR issues
const DynamicLoginForm = dynamic(() => import("@/src/components/LoginForm"), { ssr: false });
const DynamicSignupForm = dynamic(() => import("@/src/components/SignupForm"), { ssr: false });

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">PodPromo</h1>
          <p className="text-gray-600">
            AI-powered podcast clip generation
          </p>
        </div>
        
        {isLogin ? (
          <DynamicLoginForm onSwitchToSignup={() => setIsLogin(false)} />
        ) : (
          <DynamicSignupForm onSwitchToLogin={() => setIsLogin(true)} />
        )}
      </div>
    </div>
  );
}
