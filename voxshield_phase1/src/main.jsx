import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ShieldCheck, Mic, Upload, Square, Play, Pause, FileAudio,
  Activity, LockKeyhole, AlertTriangle, CheckCircle2
} from 'lucide-react';
import './styles.css';

function App() {
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState('');
  const [fileName, setFileName] = useState('');
  const [playing, setPlaying] = useState(false);
  const [status, setStatus] = useState('Ready for voice input');
  const [duration, setDuration] = useState(0);
  const mediaRecorder = useRef(null);
  const chunks = useRef([]);
  const audioRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks.current = [];
      const recorder = new MediaRecorder(stream);
      mediaRecorder.current = recorder;
      recorder.ondataavailable = e => { if (e.data.size) chunks.current.push(e.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        setFileName('Live voice recording.webm');
        setStatus('Voice captured — ready for analysis');
        stream.getTracks().forEach(t => t.stop());
      };
      recorder.start();
      setDuration(0);
      setRecording(true);
      setStatus('Recording voice…');
      timerRef.current = setInterval(() => setDuration(d => d + 1), 1000);
    } catch {
      setStatus('Microphone permission was not granted');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current?.state !== 'inactive') mediaRecorder.current.stop();
    setRecording(false);
    clearInterval(timerRef.current);
  };

  const handleFile = e => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('audio/')) {
      setStatus('Please select an audio file');
      return;
    }
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(URL.createObjectURL(file));
    setFileName(file.name);
    setStatus('Audio uploaded — ready for analysis');
    setPlaying(false);
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (playing) audioRef.current.pause();
    else audioRef.current.play();
    setPlaying(!playing);
  };

  const formatTime = s => `${String(Math.floor(s / 60)).padStart(2,'0')}:${String(s % 60).padStart(2,'0')}`;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo"><ShieldCheck size={25}/></div>
          <div><strong>VoxShield</strong><span>AI Voice Security</span></div>
        </div>
        <div className="secure"><span className="dot"></span> Secure prototype</div>
      </header>

      <main>
        <section className="hero">
          <div>
            <div className="eyebrow">REAL-TIME VOICE PROTECTION</div>
            <h1>Detect. Verify.<br/><em>Protect.</em></h1>
            <p>AI-powered protection against voice cloning and impersonation attacks.</p>
          </div>
          <div className="hero-badge">
            <LockKeyhole size={18}/>
            <span>Privacy-first<br/><b>Voice Analysis</b></span>
          </div>
        </section>

        <section className="grid">
          <div className="card input-card">
            <div className="card-head">
              <div><span className="kicker">STEP 01</span><h2>Voice Input</h2></div>
              <Mic size={22}/>
            </div>

            <div className={`dropzone ${recording ? 'recording' : ''}`}>
              <div className="mic-ring"><Mic size={28}/></div>
              <h3>{recording ? 'Listening…' : 'Provide a voice sample'}</h3>
              <p>{recording ? 'Speak clearly into your microphone' : 'Record live audio or upload an existing sample'}</p>
              <div className="actions">
                {!recording ? (
                  <button className="primary" onClick={startRecording}><Mic size={17}/> Record Voice</button>
                ) : (
                  <button className="danger" onClick={stopRecording}><Square size={16} fill="currentColor"/> Stop Recording</button>
                )}
                <label className="secondary">
                  <Upload size={17}/> Upload Audio
                  <input type="file" accept="audio/*" onChange={handleFile}/>
                </label>
              </div>
            </div>

            <div className="status">
              <span className="status-icon"><Activity size={16}/></span>
              <div><b>{status}</b><small>{recording ? `Recording time: ${formatTime(duration)}` : 'MP3, WAV, M4A and WebM supported'}</small></div>
            </div>

            {audioUrl && (
              <div className="sample">
                <div className="file-icon"><FileAudio size={20}/></div>
                <div className="sample-info">
                  <b>{fileName}</b>
                  <small>Voice sample</small>
                </div>
                <button className="play" onClick={togglePlay}>{playing ? <Pause size={18}/> : <Play size={18}/>}</button>
                <audio ref={audioRef} src={audioUrl} onEnded={() => setPlaying(false)} />
              </div>
            )}
          </div>

          <div className="card analysis-card">
            <div className="card-head">
              <div><span className="kicker">STEP 02</span><h2>AI Analysis</h2></div>
              <ShieldCheck size={22}/>
            </div>
            <div className="coming">
              <div className="scan"><div className="scan-line"></div><Activity size={38}/></div>
              <h3>Analysis Engine</h3>
              <p>Audio preprocessing, voice embeddings and deepfake detection will run here.</p>
              <div className="pipeline">
                <span>Audio</span><i>→</i><span>Features</span><i>→</i><span>AI Model</span><i>→</i><span>Risk Score</span>
              </div>
              <div className="future"><AlertTriangle size={15}/> Phase 2 AI module</div>
            </div>
          </div>
        </section>

        <section className="mini-grid">
          <div><CheckCircle2 size={20}/><span><b>Real-time detection</b><small>Designed for low-latency analysis</small></span></div>
          <div><LockKeyhole size={20}/><span><b>Secure processing</b><small>Voice data handled with privacy in mind</small></span></div>
          <div><AlertTriangle size={20}/><span><b>Threat prevention</b><small>Alerts and verification in later phases</small></span></div>
        </section>
      </main>

      <footer>VoxShield • SIH 2026 • Prototype In Progress </footer>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
