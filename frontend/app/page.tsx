"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Prediction = {
  disease: string;
  confidence: number;
  advice: string;
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

  return (
    <main className="min-h-screen bg-green-50 flex flex-col items-center px-4 py-10">
      <div className="w-full max-w-xl">
        <h1 className="text-3xl font-bold text-green-900 text-center">
          🌿 Plant Disease Detector
        </h1>
        <p className="mt-2 text-center text-green-700">
          Upload a photo of a plant leaf to check if it&apos;s healthy.
        </p>

        <div className="mt-8 bg-white rounded-2xl shadow-sm p-6">
          <label className="block">
            <span className="sr-only">Choose a leaf photo</span>
            <input
              type="file"
              accept="image/*"
              onChange={onSelect}
              className="block w-full text-sm text-gray-600 file:mr-4 file:rounded-full file:border-0 file:bg-green-600 file:px-4 file:py-2 file:text-white hover:file:bg-green-700 file:cursor-pointer cursor-pointer"
            />
          </label>

          {preview && (
            <img
              src={preview}
              alt="Selected leaf"
              className="mt-4 w-full max-h-72 object-contain rounded-lg border border-green-100"
            />
          )}

          <button
            onClick={onAnalyze}
            disabled={!file || loading}
            className="mt-4 w-full rounded-full bg-green-600 py-3 font-semibold text-white transition-colors hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Analyzing…" : "Analyze leaf"}
          </button>

          {error && (
            <p className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">
              {error}
            </p>
          )}

          {result && (
            <div className="mt-6 rounded-xl border border-green-100 bg-green-50 p-4">
              <p className="text-sm text-green-700">Diagnosis</p>
              <p className="text-xl font-bold text-green-900">{result.disease}</p>
              <p className="mt-1 text-sm text-green-700">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </p>
              <hr className="my-3 border-green-100" />
              <p className="text-sm text-green-700">Advice</p>
              <p className="mt-1 text-green-900">{result.advice}</p>
            </div>
          )}
        </div>

        <p className="mt-6 text-center text-xs text-green-600">
          Backend: {API_URL}
        </p>
      </div>
    </main>
  );
}
