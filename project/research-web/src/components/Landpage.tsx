import React, { useState, useRef } from 'react'
import { UploadCloud, CheckCircle2, AlertTriangle, Loader2, Image as ImageIcon, X } from 'lucide-react'
import './Landpage.css'

interface PredictionResult {
    prediction: string;
    confidence: number;
    heatmap: string;
}

export default function Landpage() {
    const [image, setImage] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleFile = (file: File) => {
        setImage(file);
        setPreviewUrl(URL.createObjectURL(file));
        setResult(null);
    }

    const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    }

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    }

    const resetState = () => {
        setImage(null);
        setPreviewUrl(null);
        setResult(null);
    }

    const sendToBackend = async () => {
        if (!image) return;
        setLoading(true);

        const formData = new FormData();
        formData.append("file", image);

        try {
            const res = await fetch("http://localhost:5000/predict", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();
            setResult(data);
        } catch (error) {
            console.error("Error analyzing image:", error);
            alert("Failed to connect to the backend server. Make sure it's running on port 5000.");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="dashboard">
            {!previewUrl ? (
                <div 
                    className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => inputRef.current?.click()}
                >
                    <UploadCloud className="upload-icon" size={64} />
                    <h2 className="upload-text">Upload Image for Analysis</h2>
                    <p className="upload-subtext">Drag & drop or click to browse</p>
                    <input 
                        ref={inputRef}
                        type='file' 
                        className="hidden-input" 
                        accept="image/*"
                        onChange={handleUpload} 
                    />
                </div>
            ) : (
                <div className="preview-container">
                    {!result ? (
                        <div className="glass-panel animate-fade-in" style={{ padding: '2rem', textAlign: 'center' }}>
                            <div className="image-preview-container" style={{ maxWidth: '400px', margin: '0 auto', aspectRatio: 'auto', maxHeight: '400px', backgroundColor: 'transparent' }}>
                                <img src={previewUrl} alt="Preview" className="image-preview" />
                            </div>
                            <div className="preview-actions">
                                <button className="button reset-button" onClick={resetState}>
                                    <X size={20} /> Cancel
                                </button>
                                <button className="button" onClick={sendToBackend} disabled={loading}>
                                    {loading ? <Loader2 className="animate-spin" size={20} /> : <ImageIcon size={20} />}
                                    {loading ? 'Analyzing...' : 'Run Analysis'}
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="results-grid">
                            {/* Original Image Card */}
                            <div className="glass-panel result-card">
                                <div className="result-header">
                                    <h3 className="result-title">Original Input</h3>
                                </div>
                                <div className="image-preview-container">
                                    <img src={previewUrl} alt="Original" className="image-preview" />
                                </div>
                                <button className="button reset-button" style={{ marginTop: 'auto' }} onClick={resetState}>
                                    Analyze Another Image
                                </button>
                            </div>

                            {/* Analysis Result Card */}
                            <div className={`glass-panel result-card ${result.prediction === 'Manipulated' ? 'animate-pulse-border' : ''}`}>
                                <div className="result-header">
                                    <h3 className="result-title">Analysis Result</h3>
                                    {result.prediction === 'Manipulated' ? (
                                        <div className="badge manipulated">
                                            <AlertTriangle size={16} /> Manipulated
                                        </div>
                                    ) : (
                                        <div className="badge real">
                                            <CheckCircle2 size={16} /> Authentic
                                        </div>
                                    )}
                                </div>

                                {result.prediction === 'Manipulated' ? (
                                    <div className="image-preview-container">
                                        <img src={`data:image/jpeg;base64,${result.heatmap}`} alt="Heatmap" className="image-preview" />
                                    </div>
                                ) : (
                                    <div className="image-preview-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(16, 185, 129, 0.05)' }}>
                                        <div style={{ textAlign: 'center', color: 'var(--success-color)' }}>
                                            <CheckCircle2 size={64} style={{ margin: '0 auto 1rem' }} />
                                            <p>No manipulation detected</p>
                                        </div>
                                    </div>
                                )}

                                <div className="metrics" style={{ marginTop: 'auto' }}>
                                    <div className="metric-row">
                                        <span>Confidence Score</span>
                                        <span style={{ fontWeight: 600 }}>{(result.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="confidence-bar-bg">
                                        <div 
                                            className={`confidence-bar-fill ${result.prediction === 'Manipulated' ? 'manipulated' : 'real'}`} 
                                            style={{ width: `${result.confidence * 100}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}