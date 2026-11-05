import React from "react";

/**
 * Skeleton loaders. Drawn to match the layout they stand in for, so the
 * swap-in of real content doesn't shift anything vertically (CLS ≈ 0).
 * Never used to fake progress — they only appear while the corresponding
 * real content is genuinely not yet available.
 */

export const Skeleton: React.FC<{ className?: string; rounded?: boolean }> = ({
  className = "",
  rounded = true,
}) => <div aria-hidden="true" className={`skeleton ${rounded ? "rounded-md2" : ""} ${className}`} />;

/** Mirrors HumanReviewModal layout during step1_running. */
export const ReviewSkeleton: React.FC = () => (
  <div className="panel p-5 mb-8" role="status" aria-label="Analyzing your documents">
    <Skeleton className="h-3 w-44 mb-4" />
    <div className="grid grid-cols-1 gap-4">
      <Skeleton className="h-10 w-full" />
      <Skeleton className="h-10 w-11/12" />
      <Skeleton className="h-10 w-9/12" />
      <Skeleton className="h-20 w-full" />
    </div>
    <p className="text-2xs text-ink-3 mt-3 font-mono">Reading the job posting and your resume...</p>
  </div>
);

/** Mirrors ResultsDashboard layout while step 2 streams. */
export const ResultsSkeleton: React.FC = () => (
  <div className="panel p-5 mb-8" role="status" aria-label="Preparing your documents">
    <div className="flex items-center gap-4 mb-5">
      <Skeleton className="h-20 w-20 rounded-lg2" />
      <div className="flex-1 space-y-2">
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-3 w-56" />
      </div>
    </div>
    <div className="flex gap-2 mb-4">
      <Skeleton className="h-7 w-24" />
      <Skeleton className="h-7 w-28" />
      <Skeleton className="h-7 w-20" />
      <Skeleton className="h-7 w-32" />
    </div>
    <div className="space-y-2">
      <Skeleton className="h-3 w-full" />
      <Skeleton className="h-3 w-11/12" />
      <Skeleton className="h-3 w-10/12" />
      <Skeleton className="h-3 w-full" />
      <Skeleton className="h-3 w-8/12" />
    </div>
  </div>
);
