import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ShieldCheck, Mic, Upload, Square, Play, Pause, FileAudio,
  Activity, LockKeyhole, AlertTriangle, CheckCircle2, UserRoundCheck
} from 'lucide-react';
import './styles.css';

const API = 'http://127.0.0.1:8000';

async function blobToWavFile(blob, filename) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const context = new AudioContextClass();
  try {
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    const channels = Math.min(buffer.numberOfChannels, 2);
    const length = buffer.length * channels * 2 + 44;
    const arrayBuffer = new ArrayBuffer(length);
    const view = new DataView(arrayBuffer);

    const writeString = (offset, value) => {
      for (let i = 0; i < value.length; i++) {
        view.setUint8(offset + i, value.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, length - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channels, true);
    view.setUint32(24, buffer.sampleRate, true);
    view.setUint32(28, buffer.sampleRate * channels * 2, true);
    view.setUint16(32, channels * 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length - 44, true);

    const data = [];
    for (let channel = 0; channel < channels; channel++) {
      data.push(buffer.getChannelData(channel));
    }

    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
      for (let channel = 0; channel < channels; channel++) {
        const sample = Math.max(-1, Math.min(1, data[channel][i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
        offset += 2;
      }
    }

    return new File([arrayBuffer], filename, { type: 'audio/wav' });
  } finally {
    await context.close();
  }
}

function App() {
  const [referenceFile, setReferenceFile] = useState(null);
  const [referenceUrl, setReferenceUrl] = useState('');
  const [referenceStatus, setReferenceStatus] = useState('No reference voice enrolled');
  const [registered, setRegistered] = useState(false);

  const [incomingFile, setIncomingFile] = useState(null);
  const [incomingUrl, setIncomingUrl] = useState('');
  const [incomingName, setIncomingName] = useState('');
  const [incomingStatus, setIncomingStatus] = useState('Ready for incoming voice');
  const [recordingTarget, setRecordingTarget] = useState(null);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(null);

  const [analyzing, setAnalyzing] = useState(false);
  const [enrolling, setEnrolling] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const mediaRecorder = useRef(null);
  const chunks = useRef([]);
  const timerRef = useRef(null);
  const audioRefs = useRef({});

  useEffect(() => {
    fetch(`${API}/speaker/status`)
      .then(r => r.json())
      .then(data => {
        if (data.registered) {
          setRegistered(true);
          setReferenceStatus('Reference voice is enrolled');
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (referenceUrl) URL.revokeObjectURL(referenceUrl);
    if (incomingUrl) URL.revokeObjectURL(incomingUrl);
  }, [referenceUrl, incomingUrl]);

  const startRecording = async target => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks.current = [];
      const recorder = new MediaRecorder(stream);
      mediaRecorder.current = recorder;
      setRecordingTarget(target);
      setDuration(0);
      setResult(null);
      setError('');

      recorder.ondataavailable = e => {
        if (e.data.size) chunks.current.push(e.data);
      };

      recorder.onstop = async () => {
        try {
          const blob = new Blob(chunks.current, { type: recorder.mimeType || 'audio/webm' });
          const file = await blobToWavFile(
            blob,
            target === 'reference' ? 'reference_voice.wav' : 'incoming_voice.wav'
          );
          const url = URL.createObjectURL(file);

          if (target === 'reference') {
            if (referenceUrl) URL.revokeObjectURL(referenceUrl);
            setReferenceFile(file);
            setReferenceUrl(url);
            setReferenceStatus('Reference captured — ready to enroll');
          } else {
            if (incomingUrl) URL.revokeObjectURL(incomingUrl);
            setIncomingFile(file);
            setIncomingUrl(url);
            setIncomingName(file.name);
            setIncomingStatus('Incoming voice captured — ready for analysis');
          }
        } catch (err) {
          console.error(err);
          setError('Could not convert the browser recording to WAV.');
        } finally {
          stream.getTracks().forEach(t => t.stop());
          setRecordingTarget(null);
        }
      };

      recorder.start();
      setDuration(0);
      timerRef.current = setInterval(() => setDuration(d => d + 1), 1000);

      if (target === 'reference') setReferenceStatus('Recording reference voice…');
      else setIncomingStatus('Recording incoming voice…');
    } catch {
      setError('Microphone permission was not granted.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current?.state !== 'inactive') {
      mediaRecorder.current.stop();
    }
    clearInterval(timerRef.current);
  };

  const handleFile = (e, target) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('audio/')) {
      setError('Please select an audio file.');
      return;
    }

    setResult(null);
    setError('');

    if (target === 'reference') {
      if (referenceUrl) URL.revokeObjectURL(referenceUrl);
      setReferenceFile(file);
      setReferenceUrl(URL.createObjectURL(file));
      setReferenceStatus('Reference audio selected — ready to enroll');
    } else {
      if (incomingUrl) URL.revokeObjectURL(incomingUrl);
      setIncomingFile(file);
      setIncomingUrl(URL.createObjectURL(file));
      setIncomingName(file.name);
      setIncomingStatus('Incoming audio selected — ready for analysis');
    }
  };

  const enrollReference = async () => {
    if (!referenceFile) {
      setError('Record or upload a reference voice first.');
      return;
    }

    setEnrolling(true);
    setError('');
    setReferenceStatus('Enrolling reference voice…');

    try {
      const formData = new FormData();
      formData.append('file', referenceFile);

      const response = await fetch(`${API}/speaker/enroll`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Enrollment failed');

      setRegistered(true);
      setReferenceStatus('Reference voice enrolled successfully');
    } catch (err) {
      setError(err.message || 'Could not enroll the reference voice.');
      setReferenceStatus('Reference enrollment failed');
    } finally {
      setEnrolling(false);
    }
  };

  const analyzeVoice = async () => {
    if (!registered) {
      setError('Enroll a reference voice before analyzing an incoming voice.');
      return;
    }
    if (!incomingFile) {
      setError('Record or upload an incoming voice first.');
      return;
    }

    setAnalyzing(true);
    setError('');
    setResult(null);
    setIncomingStatus('Analyzing incoming voice…');

    try {
      const formData = new FormData();
      formData.append('file', incomingFile);

      const response = await fetch(`${API}/analyze`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Analysis request failed');

      setResult(data);
      setIncomingStatus('Analysis complete');
    } catch (err) {
      console.error(err);
      setError(err.message || 'Could not connect to the VoxShield analysis server.');
      setIncomingStatus('Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const togglePlay = target => {
    const audio = audioRefs.current[target];
    if (!audio) return;
    if (playing === target) {
      audio.pause();
      setPlaying(null);
    } else {
      Object.values(audioRefs.current).forEach(a => a?.pause());
      audio.play();
      setPlaying(target);
    }
  };

  const formatTime = s =>
    `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

  const isRecording = recordingTarget !== null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo"><ShieldCheck size={25}/></div>
          <div><strong>VoxShield</strong><span>AI Voice Security</span></div>
        </div>
        <div className="secure"><span className="dot"></span>Secure prototype</div>
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
              <div><span className="kicker">STEP 01</span><h2>Speaker Enrollment</h2></div>
              <UserRoundCheck size={22}/>
            </div>

            <div className={`dropzone ${recordingTarget === 'reference' ? 'recording' : ''}`}>
              <div className="mic-ring"><Mic size={28}/></div>
              <h3>{recordingTarget === 'reference' ? 'Listening…' : 'Enroll a reference voice'}</h3>
              <p>Record a clean sample of the person whose identity should be verified.</p>
              <div className="actions">
                {!recordingTarget ? (
                  <button className="primary" onClick={() => startRecording('reference')}>
                    <Mic size={17}/> Record Reference
                  </button>
                ) : recordingTarget === 'reference' ? (
                  <button className="danger" onClick={stopRecording}>
                    <Square size={16} fill="currentColor"/> Stop Recording
                  </button>
                ) : null}
                {!isRecording && (
                  <label className="secondary">
                    <Upload size={17}/> Upload Reference
                    <input type="file" accept="audio/*" onChange={e => handleFile(e, 'reference')}/>
                  </label>
                )}
              </div>
            </div>

            <div className="status">
              <span className="status-icon"><Activity size={16}/></span>
              <div><b>{referenceStatus}</b><small>{recordingTarget === 'reference' ? `Recording time: ${formatTime(duration)}` : 'Reference voice is stored as an ECAPA-TDNN embedding'}</small></div>
            </div>

            {referenceUrl && (
              <div className="sample">
                <div className="file-icon"><FileAudio size={20}/></div>
                <div className="sample-info"><b>Reference voice</b><small>Audio sample</small></div>
                <button className="play" onClick={() => togglePlay('reference')}>
                  {playing === 'reference' ? <Pause size={18}/> : <Play size={18}/>}
                </button>
                <audio ref={el => { audioRefs.current.reference = el; }} src={referenceUrl} onEnded={() => setPlaying(null)}/>
              </div>
            )}

            {referenceFile && !recordingTarget && (
              <button className="primary" onClick={enrollReference} disabled={enrolling}
                style={{width:'100%', marginTop:'16px', justifyContent:'center'}}>
                <UserRoundCheck size={18}/>
                {enrolling ? 'Enrolling Voice…' : registered ? 'Re-enroll Reference Voice' : 'Enroll Reference Voice'}
              </button>
            )}

            <div className="card-head" style={{marginTop:'30px'}}>
              <div><span className="kicker">STEP 02</span><h2>Incoming Voice</h2></div>
              <Mic size={22}/>
            </div>

            <div className={`dropzone ${recordingTarget === 'incoming' ? 'recording' : ''}`}>
              <div className="mic-ring"><Mic size={28}/></div>
              <h3>{recordingTarget === 'incoming' ? 'Listening…' : 'Provide incoming voice'}</h3>
              <p>Record or upload the voice you want VoxShield to analyze.</p>
              <div className="actions">
                {!recordingTarget ? (
                  <button className="primary" onClick={() => startRecording('incoming')}>
                    <Mic size={17}/> Record Incoming
                  </button>
                ) : recordingTarget === 'incoming' ? (
                  <button className="danger" onClick={stopRecording}>
                    <Square size={16} fill="currentColor"/> Stop Recording
                  </button>
                ) : null}
                {!isRecording && (
                  <label className="secondary">
                    <Upload size={17}/> Upload Audio
                    <input type="file" accept="audio/*" onChange={e => handleFile(e, 'incoming')}/>
                  </label>
                )}
              </div>
            </div>

            <div className="status">
              <span className="status-icon"><Activity size={16}/></span>
              <div><b>{incomingStatus}</b><small>{recordingTarget === 'incoming' ? `Recording time: ${formatTime(duration)}` : 'MP3, WAV, M4A and WebM supported'}</small></div>
            </div>

            {incomingUrl && (
              <div className="sample">
                <div className="file-icon"><FileAudio size={20}/></div>
                <div className="sample-info"><b>{incomingName}</b><small>Incoming voice sample</small></div>
                <button className="play" onClick={() => togglePlay('incoming')}>
                  {playing === 'incoming' ? <Pause size={18}/> : <Play size={18}/>}
                </button>
                <audio ref={el => { audioRefs.current.incoming = el; }} src={incomingUrl} onEnded={() => setPlaying(null)}/>
              </div>
            )}

            {incomingFile && !recordingTarget && (
              <button className="primary" onClick={analyzeVoice} disabled={analyzing || !registered}
                style={{width:'100%', marginTop:'16px', justifyContent:'center'}}>
                <ShieldCheck size={18}/>
                {analyzing ? 'Analyzing Voice…' : registered ? 'Analyze Voice' : 'Enroll Reference First'}
              </button>
            )}

            {error && (
              <div style={{marginTop:'15px', padding:'12px', borderRadius:'10px', background:'#fff1f1', color:'#b42318'}}>
                {error}
              </div>
            )}
          </div>

          <div className="card analysis-card">
            <div className="card-head">
              <div><span className="kicker">STEP 03</span><h2>AI Analysis</h2></div>
              <ShieldCheck size={22}/>
            </div>

            {!result ? (
              <div className="coming">
                <div className="scan"><div className="scan-line"></div><Activity size={38}/></div>
                <h3>{analyzing ? 'Analyzing Voice…' : 'Analysis Engine'}</h3>
                <p>Speaker verification is now connected. Deepfake detection will be added from the AI module.</p>
                <div className="pipeline"><span>Audio</span><i>→</i><span>ECAPA</span><i>→</i><span>Deepfake AI</span><i>→</i><span>Risk Score</span></div>
                <div className="future"><AlertTriangle size={15}/> Member 1 deepfake module pending</div>
              </div>
            ) : (
              <div style={{padding:'10px 0'}}>
                <h3>Analysis Result</h3>
                <div style={{marginTop:'18px'}}>
                  <p><b>Authenticity Score:</b> {result.authenticity_score}%</p>
                  <p><b>AI / Spoof Probability:</b> {result.spoof_probability * 100}% <small>(temporary placeholder)</small></p>
                  <p><b>Speaker Similarity:</b> {(result.speaker_similarity * 100).toFixed(2)}%</p>
                  <p><b>Speaker Match:</b> {result.speaker_match ? 'YES' : 'NO'}</p>
                  {result.audio_features && (
                    <div style={{marginTop:'16px', padding:'12px 14px', borderRadius:'10px', background:'#f7faf9'}}>
                      <p style={{margin:'0 0 6px'}}><b>Audio Processing:</b></p>
                      <p style={{margin:'4px 0', fontSize:'14px'}}>Mono • {result.audio_features.sample_rate / 1000} kHz • {result.audio_features.duration_seconds}s</p>
                      <p style={{margin:'4px 0', fontSize:'14px'}}>MFCC: {result.audio_features.mfcc_dimensions?.[1] || 40} coefficients • Log-Mel: {result.audio_features.mel_dimensions?.[1] || 64} bands</p>
                      <p style={{margin:'4px 0', fontSize:'13px', opacity:.7}}>ECAPA-TDNN speaker embedding extracted separately</p>
                    </div>
                  )}
                  <p><b>Risk Level:</b> {result.risk_level}</p>
                  <p><b>Decision:</b> {result.decision}</p>
                </div>
              </div>
            )}
          </div>
        </section>

        <section className="mini-grid">
          <div><CheckCircle2 size={20}/><span><b>Real-time detection</b><small>Designed for low-latency analysis</small></span></div>
          <div><LockKeyhole size={20}/><span><b>Secure processing</b><small>Voice data handled with privacy in mind</small></span></div>
          <div><AlertTriangle size={20}/><span><b>Threat prevention</b><small>Alerts and verification in later phases</small></span></div>
        </section>
      </main>

      <footer>VoxShield • SIH 2026 • Prototype In Progress</footer>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
