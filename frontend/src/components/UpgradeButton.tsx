"use client";
import { useState } from "react";

interface UpgradeButtonProps {
  userId: string;
  currentPlan: string;
  onUpgrade?: () => void;
}

export default function UpgradeButton({ userId, currentPlan, onUpgrade }: UpgradeButtonProps) {
  const [loading, setLoading] = useState(false);

  const upgrade = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/paddle/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, plan: "pro" }),
      });
      
      const data = await res.json();
      
      if (data.url) {
        // Redirect to Paddle checkout
        window.location.href = data.url;
      } else {
        if (process.env.NODE_ENV === "development") {
          console.error("No checkout URL received");
        }
        setLoading(false);
      }
    } catch (error) {
      if (process.env.NODE_ENV === "development") {
        console.error("Upgrade failed:", error);
      }
      setLoading(false);
    }
  };

  // Don't show upgrade button if user is already on Pro plan
  if (currentPlan === "pro") {
    return (
      <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-2 rounded-lg">
        ✓ Pro Plan Active
      </div>
    );
  }

  return (
    <button
      onClick={upgrade}
      disabled={loading}
      className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-6 py-3 rounded-lg font-medium transition-colors"
    >
      {loading ? "Redirecting..." : "Upgrade to Pro ($12.99/mo)"}
    </button>
  );
}
