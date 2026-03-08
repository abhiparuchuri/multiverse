"use client";

import { useCallback, useState } from "react";
import { useAppState } from "@/lib/store";
import { uploadCSV } from "@/lib/api";
import { Upload, FileSpreadsheet, Loader2 } from "lucide-react";

export function ImportPhase() {
  const { setPhase, setSessionId, setDataProfile, setColumns, setChatMessages, setCompletedPhases, completedPhases } = useAppState();
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.endsWith(".csv")) {
        setError("Please upload a CSV file");
        return;
      }
      setError(null);
      setFileName(file.name);
      setUploading(true);

      try {
        const data = await uploadCSV(file);
        setSessionId(data.session_id);
        setDataProfile(data.profile);
        setColumns(data.profile.column_profiles.map((c: { name: string }) => c.name));
        setChatMessages(data.chat_messages || []);
        setCompletedPhases([...completedPhases, "import"]);
        setPhase("variables");
      } catch (e) {
        setError("Failed to upload file. Is the backend running?");
      } finally {
        setUploading(false);
      }
    },
    [setPhase, setSessionId, setDataProfile, setColumns, setChatMessages, setCompletedPhases, completedPhases]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="w-full max-w-lg">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-semibold mb-2">Import Your Dataset</h2>
          <p className="text-muted-foreground">
            Upload a CSV file to begin your analysis validation
          </p>
        </div>

        <label
          onDrop={onDrop}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          className={`
            relative flex flex-col items-center justify-center
            w-full h-64 rounded-xl border-2 border-dashed
            transition-all cursor-pointer
            ${
              dragOver
                ? "border-foreground bg-accent/50"
                : "border-border hover:border-muted-foreground hover:bg-accent/30"
            }
            ${uploading ? "pointer-events-none opacity-60" : ""}
          `}
        >
          <input
            type="file"
            accept=".csv"
            onChange={onFileSelect}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          {uploading ? (
            <>
              <Loader2 className="w-10 h-10 text-muted-foreground animate-spin mb-4" />
              <p className="text-sm text-muted-foreground">
                Analyzing {fileName}...
              </p>
            </>
          ) : fileName ? (
            <>
              <FileSpreadsheet className="w-10 h-10 text-muted-foreground mb-4" />
              <p className="text-sm text-foreground font-medium">{fileName}</p>
              <p className="text-xs text-muted-foreground mt-1">
                Processing...
              </p>
            </>
          ) : (
            <>
              <Upload className="w-10 h-10 text-muted-foreground mb-4" />
              <p className="text-sm text-foreground font-medium">
                Drop your CSV here or click to browse
              </p>
              <p className="text-xs text-muted-foreground mt-2">
                Supports .csv files
              </p>
            </>
          )}
        </label>

        {error && (
          <div className="mt-4 p-3 rounded-lg bg-destructive/10 border border-destructive/20">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}
