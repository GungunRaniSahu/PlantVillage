"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Prediction = {
  disease: string;
  confidence: number;
  advice: string;
  heatmap?: string | null;
};

export default function Home() {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function onSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(f ? URL.createObjectURL(f) : null);
    e.target.value = ""; // allow re-selecting the same file later
  }

  function onReset() {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }

  async function onAnalyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? `Request failed (${res.status})`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const confidence = result ? Math.round(result.confidence * 100) : 0;

  return (
    <main className="min-h-screen bg-[#06120d] text-emerald-50">
      <div className="mx-auto w-full max-w-3xl px-5 py-10">
        <header className="text-center">
          <span className="inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-[11px] font-medium text-emerald-300">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-lime-300" />
            15 classes · 3 crops · 92% accuracy
          </span>
          <h1 className="mt-3 text-3xl font-black tracking-tight text-lime-200">
            Leaf Doctor
          </h1>
          <p className="mt-1.5 text-sm text-emerald-200/60">
            Upload a leaf photo to get a diagnosis and a Grad-CAM view.
          </p>
        </header>

        <div className="mt-7 grid items-start gap-4 md:grid-cols-2">
          {/* upload panel */}
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <label className="group flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-emerald-400/30 bg-emerald-950/30 px-4 py-7 text-center transition-all hover:border-lime-300/60 hover:bg-emerald-900/30">
              <input
                type="file"
                accept="image/*"
                onChange={onSelect}
                className="hidden"
              />
              {preview ? (
                <div className="relative w-full">
                  <img
                    src={preview}
                    alt="Selected leaf"
                    className="max-h-56 w-full rounded-lg object-contain"
                  />
                  <div className="absolute inset-0 flex items-center justify-center rounded-lg opacity-0 transition group-hover:bg-emerald-950/60 group-hover:opacity-100">
                    <span className="text-xs font-medium text-emerald-100">
                      Click to change photo
                    </span>
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-lime-300 text-xl text-emerald-950 transition-transform group-hover:scale-110">
                    ↑
                  </div>
                  <span className="mt-2 text-sm font-medium text-emerald-100">
                    Click to upload a leaf
                  </span>
                  <span className="mt-0.5 text-xs text-emerald-200/50">
                    JPG or PNG
                  </span>
                </>
              )}
            </label>

            {file && (
              <p className="mt-2 truncate text-center text-xs text-emerald-200/50">
                {file.name}
              </p>
            )}

            <button
              onClick={onAnalyze}
              disabled={!file || loading}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-full bg-lime-300 py-2.5 text-sm font-bold text-emerald-950 shadow-lg shadow-emerald-500/20 transition-all hover:bg-lime-200 hover:shadow-emerald-400/40 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
            >
              {loading ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-emerald-950/30 border-t-emerald-950" />
                  Analyzing…
                </>
              ) : (
                "Diagnose leaf"
              )}
            </button>

            {error && (
              <p className="mt-3 rounded-lg border border-red-400/20 bg-red-500/10 px-3 py-2 text-sm text-red-300">
                {error}
              </p>
            )}

            {(file || result) && (
              <button
                onClick={onReset}
                className="mt-2 w-full text-xs text-emerald-200/50 transition-colors hover:text-emerald-200"
              >
                Start over
              </button>
            )}
          </div>

          {/* result panel */}
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            {result ? (
              <div className="animate-rise space-y-5">
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-widest text-emerald-300/60">
                    Diagnosis
                  </p>
                  <p className="mt-1 text-xl font-bold text-lime-200">
                    {result.disease}
                  </p>
                  <div className="mt-2.5 flex items-center gap-3">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full bg-lime-400 transition-all duration-700"
                        style={{ width: `${confidence}%` }}
                      />
                    </div>
                    <span className="text-sm font-semibold tabular-nums text-emerald-200">
                      {confidence}%
                    </span>
                  </div>
                </div>

                {result.heatmap && (
                  <div>
                    <p className="text-[11px] font-medium uppercase tracking-widest text-emerald-300/60">
                      Where the model looked
                    </p>
                    <div className="mt-2 grid grid-cols-2 gap-2">
                      {preview && (
                        <figure>
                          <img
                            src={preview}
                            alt="Original leaf"
                            className="w-full rounded-lg border border-white/10 object-contain"
                          />
                          <figcaption className="mt-1 text-center text-xs text-emerald-200/50">
                            Original
                          </figcaption>
                        </figure>
                      )}
                      <figure>
                        <img
                          src={result.heatmap}
                          alt="Grad-CAM heatmap"
                          className="w-full rounded-lg border border-white/10 object-contain"
                        />
                        <figcaption className="mt-1 text-center text-xs text-emerald-200/50">
                          Grad-CAM
                        </figcaption>
                      </figure>
                    </div>
                  </div>
                )}

                {result.advice && (
                  <div>
                    <p className="text-[11px] font-medium uppercase tracking-widest text-emerald-300/60">
                      Advice
                    </p>
                    <p className="mt-1 text-sm leading-relaxed text-emerald-100/80">
                      {result.advice}
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex h-full min-h-[180px] flex-col items-center justify-center text-center">
                <p className="text-sm text-emerald-200/40">
                  Your diagnosis will appear here
                </p>
              </div>
            )}
          </div>
        </div>

        <p className="mt-8 text-center text-[11px] text-emerald-200/30">
          {API_URL}
        </p>
      </div>
    </main>
  );
}
