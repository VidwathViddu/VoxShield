import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ShieldCheck,
  Mic,
  Upload,
  Square,
  Play,
  Pause,
  FileAudio,
  Activity,
  LockKeyhole,
  AlertTriangle,
  CheckCircle2,
  UserRoundCheck
} from 'lucide-react';
import './styles.css';

const API = 'http://127.0.0.1:8000';

async function blobToWavFile(blob, filename) {
  const AudioContextClass =
    window.AudioContext || window.webkitAudioContext;

  const context = new AudioContextClass();

  try {
    const buffer = await context.decodeAudioData(
      await blob.arrayBuffer()
    );

    const channels = Math.min(buffer.numberOfChannels, 2);
    const length = buffer.length * channels * 2 + 44;

    const arrayBuffer = new ArrayBuffer(length);
    const view = new DataView(arrayBuffer);

    const writeString = (offset, value) => {
      for (let i = 0; i < value.length; i++) {
        view.setUint8(
          offset + i,
          value.charCodeAt(i)
        );
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
    view.setUint32(
      28,
      buffer.sampleRate * channels * 2,
      true
    );
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
        const sample = Math.max(
          -1,
          Math.min(1, data[channel][i])
        );

        view.setInt16(
          offset,
          sample < 0
            ? sample * 0x8000
            : sample * 0x7fff,
          true
        );

        offset += 2;
      }
    }

    return new File(
      [arrayBuffer],
      filename,
      { type: 'audio/wav' }
    );
  } finally {
    await context.close();
  }
}

function App() {
  const [activePage, setActivePage] = useState('dashboard');

  const [referenceFile, setReferenceFile] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);

  const [referenceUrl, setReferenceUrl] = useState('');
  const [referenceStatus, setReferenceStatus] =
    useState('No reference voice enrolled');

  const [referenceCount, setReferenceCount] = useState(0);
  const [registered, setRegistered] = useState(false);

  const [incomingFile, setIncomingFile] = useState(null);
  const [incomingUrl, setIncomingUrl] = useState('');
  const [incomingName, setIncomingName] = useState('');

  const [incomingStatus, setIncomingStatus] =
    useState('Ready for incoming voice');

  const [recordingTarget, setRecordingTarget] =
    useState(null);

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


  /* ================================
     LOAD SPEAKER STATUS
     ================================ */

  useEffect(() => {
    fetch(`${API}/speaker/status`)
      .then(response => response.json())
      .then(data => {
        if (data.registered) {
          setRegistered(true);

          const count =
            data.reference_count || 1;

          setReferenceCount(count);

          setReferenceStatus(
            count > 1
              ? `Speaker profile active — ${count} reference recordings`
              : 'Speaker profile active — 1 reference recording'
          );
        } else {
          setRegistered(false);
          setReferenceCount(0);
          setReferenceStatus(
            'No reference voice enrolled'
          );
        }
      })
      .catch(() => {});
  }, []);


  /* ================================
     CLEANUP
     ================================ */

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }

      if (referenceUrl) {
        URL.revokeObjectURL(referenceUrl);
      }

      if (incomingUrl) {
        URL.revokeObjectURL(incomingUrl);
      }
    };
  }, [referenceUrl, incomingUrl]);


  /* ================================
     RECORDING
     ================================ */

  const startRecording = async target => {
    try {
      const stream =
        await navigator.mediaDevices.getUserMedia({
          audio: true
        });

      chunks.current = [];

      const recorder = new MediaRecorder(stream);

      mediaRecorder.current = recorder;

      setRecordingTarget(target);
      setDuration(0);
      setResult(null);
      setError('');

      recorder.ondataavailable = event => {
        if (event.data.size) {
          chunks.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        try {
          const blob = new Blob(
            chunks.current,
            {
              type:
                recorder.mimeType ||
                'audio/webm'
            }
          );

          const file = await blobToWavFile(
            blob,
            target === 'reference'
              ? 'reference_voice.wav'
              : 'incoming_voice.wav'
          );

          const url =
            URL.createObjectURL(file);

          const downloadUrl =
            URL.createObjectURL(file);

          const a =
            document.createElement('a');

          a.href = downloadUrl;

          a.download =
            target === 'reference'
              ? 'reference_voice.wav'
              : 'incoming_voice.wav';

          a.click();

          URL.revokeObjectURL(
            downloadUrl
          );

          if (target === 'reference') {
            if (referenceUrl) {
              URL.revokeObjectURL(
                referenceUrl
              );
            }

            setReferenceFile(file);
            setReferenceUrl(url);

            setReferenceStatus(
              'Reference captured — ready to enroll'
            );
          } else {
            if (incomingUrl) {
              URL.revokeObjectURL(
                incomingUrl
              );
            }

            setIncomingFile(file);
            setIncomingUrl(url);
            setIncomingName(file.name);

            setIncomingStatus(
              'Incoming voice captured — ready for analysis'
            );
          }
        } catch (err) {
          console.error(err);

          setError(
            'Could not convert the browser recording to WAV.'
          );
        } finally {
          stream
            .getTracks()
            .forEach(track =>
              track.stop()
            );

          setRecordingTarget(null);
        }
      };

      recorder.start();

      setDuration(0);

      timerRef.current =
        setInterval(
          () =>
            setDuration(
              current => current + 1
            ),
          1000
        );

      if (target === 'reference') {
        setReferenceStatus(
          'Recording reference voice…'
        );
      } else {
        setIncomingStatus(
          'Recording incoming voice…'
        );
      }

    } catch {
      setError(
        'Microphone permission was not granted.'
      );
    }
  };


  const stopRecording = () => {
    if (
      mediaRecorder.current?.state !==
      'inactive'
    ) {
      mediaRecorder.current.stop();
    }

    clearInterval(timerRef.current);
  };


  /* ================================
     FILE UPLOAD
     ================================ */

  const handleFile = (event, target) => {
    const file =
      event.target.files?.[0];

    if (!file) return;

    if (!file.type.startsWith('audio/')) {
      setError(
        'Please select an audio file.'
      );
      return;
    }

    setResult(null);
    setError('');

    if (target === 'reference') {
      if (referenceUrl) {
        URL.revokeObjectURL(
          referenceUrl
        );
      }

      setReferenceFile(file);

      setReferenceUrl(
        URL.createObjectURL(file)
      );

      setReferenceStatus(
        'Reference audio selected — ready to enroll'
      );

    } else {
      if (incomingUrl) {
        URL.revokeObjectURL(
          incomingUrl
        );
      }

      setIncomingFile(file);

      setIncomingUrl(
        URL.createObjectURL(file)
      );

      setIncomingName(file.name);

      setIncomingStatus(
        'Incoming audio selected — ready for analysis'
      );
    }
  };


  /* ================================
     ENROLL REFERENCE
     ================================ */

  const enrollReference = async () => {
    if (!referenceFile) {
      setError(
        'Record or upload a reference voice first.'
      );
      return;
    }

    setEnrolling(true);
    setError('');

    setReferenceStatus(
      'Enrolling reference voice…'
    );

    try {
      const formData =
        new FormData();

      formData.append(
        'file',
        referenceFile
      );

      const response =
        await fetch(
          `${API}/speaker/enroll`,
          {
            method: 'POST',
            body: formData
          }
        );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
          'Enrollment failed'
        );
      }

      setRegistered(true);

      const count =
        data.reference_count || 1;

      setReferenceCount(count);

      setReferenceStatus(
        count > 1
          ? `Speaker profile updated — ${count} reference recordings`
          : 'Speaker profile active — 1 reference recording'
      );

    } catch (err) {
      setError(
        err.message ||
        'Could not enroll the reference voice.'
      );

      setReferenceStatus(
        'Reference enrollment failed'
      );

    } finally {
      setEnrolling(false);
    }
  };


  /* ================================
     RESET SPEAKER PROFILE
     ================================ */

  const resetSpeakerProfile = async () => {
    const confirmed =
      window.confirm(
        'Reset the current speaker profile?\n\nThis will remove all enrolled reference recordings. Existing analysis history will NOT be deleted.'
      );

    if (!confirmed) return;

    setError('');

    try {
      const response =
        await fetch(
          `${API}/speaker/reset`,
          {
            method: 'POST'
          }
        );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
          'Could not reset speaker profile.'
        );
      }

      setRegistered(false);
      setReferenceCount(0);
      setReferenceFile(null);

      if (referenceUrl) {
        URL.revokeObjectURL(
          referenceUrl
        );

        setReferenceUrl('');
      }

      setReferenceStatus(
        'No reference voice enrolled'
      );

    } catch (err) {
      console.error(err);

      setError(
        err.message ||
        'Could not reset the speaker profile.'
      );
    }
  };


  /* ================================
     ANALYZE VOICE
     ================================ */

  const analyzeVoice = async () => {
    if (!registered) {
      setError(
        'Enroll a reference voice before analyzing an incoming voice.'
      );
      return;
    }

    if (!incomingFile) {
      setError(
        'Record or upload an incoming voice first.'
      );
      return;
    }

    setAnalyzing(true);
    setError('');
    setResult(null);

    setIncomingStatus(
      'Analyzing incoming voice…'
    );

    try {
      const formData =
        new FormData();

      formData.append(
        'file',
        incomingFile
      );

      const response =
        await fetch(
          `${API}/analyze`,
          {
            method: 'POST',
            body: formData
          }
        );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
          'Analysis request failed'
        );
      }

      setResult(data);

      setAnalysisHistory(
        previous => [
          {
            id: Date.now(),

            time:
              new Date().toLocaleTimeString(
                [],
                {
                  hour: '2-digit',
                  minute: '2-digit'
                }
              ),

            result: data
          },

          ...previous
        ].slice(0, 8)
      );

      setIncomingStatus(
        'Analysis complete'
      );

    } catch (err) {
      console.error(err);

      setError(
        err.message ||
        'Could not connect to the VoxShield analysis server.'
      );

      setIncomingStatus(
        'Analysis failed'
      );

    } finally {
      setAnalyzing(false);
    }
  };


  /* ================================
     AUDIO PLAYER
     ================================ */

  const togglePlay = target => {
    const audio =
      audioRefs.current[target];

    if (!audio) return;

    if (playing === target) {
      audio.pause();
      setPlaying(null);

    } else {
      Object.values(
        audioRefs.current
      ).forEach(
        audioElement =>
          audioElement?.pause()
      );

      audio.play();

      setPlaying(target);
    }
  };


  const formatTime = seconds =>
    `${String(
      Math.floor(seconds / 60)
    ).padStart(2, '0')}:${String(
      seconds % 60
    ).padStart(2, '0')}`;


  const isRecording =
    recordingTarget !== null;


  /* ================================
     RENDER
     ================================ */

  return (
    <div className="app">

      {/* TOP NAVIGATION */}

      <header className="topbar">

        <div className="brand">

          <div className="logo">
            <ShieldCheck size={25} />
          </div>

          <div>
            <strong>VoxShield</strong>
            <span>AI Voice Security</span>
          </div>

        </div>


        <nav className="nav">

          <button
            className={
              activePage === 'dashboard'
                ? 'nav-active'
                : ''
            }
            onClick={() =>
              setActivePage('dashboard')
            }
          >
            Dashboard
          </button>


          <button
            className={
              activePage === 'analysis'
                ? 'nav-active'
                : ''
            }
            onClick={() =>
              setActivePage('analysis')
            }
          >
            Voice Analysis
          </button>


          <button
            className={
              activePage === 'history'
                ? 'nav-active'
                : ''
            }
            onClick={() =>
              setActivePage('history')
            }
          >
            History
          </button>

        </nav>


        <div className="secure">
          <span className="dot"></span>
          System Operational
        </div>

      </header>


      {/* ================================
          VOICE ANALYSIS PAGE
          ================================ */}

      {activePage === 'analysis' && (
        <>

          <section className="hero">

            <div>
              <div className="eyebrow">
                REAL-TIME VOICE PROTECTION
              </div>

              <h1>
                Detect. Verify.
                <br />
                <em>Protect.</em>
              </h1>

              <p>
                AI-powered protection against
                voice cloning and impersonation attacks.
              </p>
            </div>


            <div className="hero-badge">

              <LockKeyhole size={18} />

              <span>
                Privacy-first
                <br />
                <b>Voice Analysis</b>
              </span>

            </div>

          </section>


          <section className="grid">

            {/* STEP 01 + STEP 02 */}

            <div className="card input-card">

              <div className="card-head">

                <div>
                  <span className="kicker">
                    STEP 01
                  </span>

                  <h2>
                    Speaker Enrollment
                  </h2>
                </div>

                <UserRoundCheck size={22} />

              </div>


              <div
                className={`dropzone ${
                  recordingTarget === 'reference'
                    ? 'recording'
                    : ''
                }`}
              >

                <div className="mic-ring">
                  <Mic size={28} />
                </div>

                <h3>
                  {recordingTarget === 'reference'
                    ? 'Listening…'
                    : 'Enroll a reference voice'}
                </h3>

                <p>
                  Record a clean sample of the
                  person whose identity should be verified.
                </p>


                <div className="actions">

                  {!recordingTarget ? (

                    <button
                      className="primary"
                      onClick={() =>
                        startRecording('reference')
                      }
                    >
                      <Mic size={17} />
                      Record Reference
                    </button>

                  ) : recordingTarget === 'reference' ? (

                    <button
                      className="danger"
                      onClick={stopRecording}
                    >
                      <Square
                        size={16}
                        fill="currentColor"
                      />
                      Stop Recording
                    </button>

                  ) : null}


                  {!isRecording && (

                    <label className="secondary">

                      <Upload size={17} />

                      Upload Reference

                      <input
                        type="file"
                        accept="audio/*"
                        onChange={event =>
                          handleFile(
                            event,
                            'reference'
                          )
                        }
                      />

                    </label>

                  )}

                </div>

              </div>


              <div className="status">

                <span className="status-icon">
                  <Activity size={16} />
                </span>

                <div>

                  <b>
                    {referenceStatus}
                  </b>

                  <small>
                    {recordingTarget === 'reference'
                      ? `Recording time: ${formatTime(duration)}`
                      : referenceCount > 1
                        ? `ECAPA profile built from ${referenceCount} reference recordings`
                        : 'ECAPA speaker profile is ready'}
                  </small>

                </div>

              </div>


              {referenceUrl && (

                <div className="sample">

                  <div className="file-icon">
                    <FileAudio size={20} />
                  </div>

                  <div className="sample-info">
                    <b>Reference voice</b>
                    <small>Audio sample</small>
                  </div>

                  <button
                    className="play"
                    onClick={() =>
                      togglePlay('reference')
                    }
                  >
                    {playing === 'reference'
                      ? <Pause size={18} />
                      : <Play size={18} />}
                  </button>

                  <audio
                    ref={element => {
                      audioRefs.current.reference =
                        element;
                    }}
                    src={referenceUrl}
                    onEnded={() =>
                      setPlaying(null)
                    }
                  />

                </div>

              )}


              {referenceFile &&
                !recordingTarget && (

                  <button
                    className="primary"
                    onClick={enrollReference}
                    disabled={enrolling}
                    style={{
                      width: '100%',
                      marginTop: '16px',
                      justifyContent: 'center'
                    }}
                  >
                    <UserRoundCheck size={18} />

                    {enrolling
                      ? 'Enrolling Voice…'
                      : registered
                        ? 'Re-enroll Reference Voice'
                        : 'Enroll Reference Voice'}
                  </button>

                )}


              {registered && (

                <button
                  className="reset-profile-btn"
                  onClick={
                    resetSpeakerProfile
                  }
                >
                  Reset Speaker Profile
                </button>

              )}


              {/* STEP 02 */}

              <div
                className="card-head"
                style={{
                  marginTop: '30px'
                }}
              >

                <div>
                  <span className="kicker">
                    STEP 02
                  </span>

                  <h2>
                    Incoming Voice
                  </h2>
                </div>

                <Mic size={22} />

              </div>


              <div
                className={`dropzone ${
                  recordingTarget === 'incoming'
                    ? 'recording'
                    : ''
                }`}
              >

                <div className="mic-ring">
                  <Mic size={28} />
                </div>

                <h3>
                  {recordingTarget === 'incoming'
                    ? 'Listening…'
                    : 'Provide incoming voice'}
                </h3>

                <p>
                  Record or upload the voice
                  you want VoxShield to analyze.
                </p>


                <div className="actions">

                  {!recordingTarget ? (

                    <button
                      className="primary"
                      onClick={() =>
                        startRecording('incoming')
                      }
                    >
                      <Mic size={17} />
                      Record Incoming
                    </button>

                  ) : recordingTarget === 'incoming' ? (

                    <button
                      className="danger"
                      onClick={stopRecording}
                    >
                      <Square
                        size={16}
                        fill="currentColor"
                      />
                      Stop Recording
                    </button>

                  ) : null}


                  {!isRecording && (

                    <label className="secondary">

                      <Upload size={17} />

                      Upload Audio

                      <input
                        type="file"
                        accept="audio/*"
                        onChange={event =>
                          handleFile(
                            event,
                            'incoming'
                          )
                        }
                      />

                    </label>

                  )}

                </div>

              </div>


              <div className="status">

                <span className="status-icon">
                  <Activity size={16} />
                </span>

                <div>

                  <b>
                    {incomingStatus}
                  </b>

                  <small>
                    {recordingTarget === 'incoming'
                      ? `Recording time: ${formatTime(duration)}`
                      : 'WAV and compatible audio files supported'}
                  </small>

                </div>

              </div>


              {incomingUrl && (

                <div className="sample">

                  <div className="file-icon">
                    <FileAudio size={20} />
                  </div>

                  <div className="sample-info">
                    <b>{incomingName}</b>
                    <small>
                      Incoming voice sample
                    </small>
                  </div>

                  <button
                    className="play"
                    onClick={() =>
                      togglePlay('incoming')
                    }
                  >
                    {playing === 'incoming'
                      ? <Pause size={18} />
                      : <Play size={18} />}
                  </button>

                  <audio
                    ref={element => {
                      audioRefs.current.incoming =
                        element;
                    }}
                    src={incomingUrl}
                    onEnded={() =>
                      setPlaying(null)
                    }
                  />

                </div>

              )}


              {incomingFile &&
                !recordingTarget && (

                  <button
                    className="primary"
                    onClick={analyzeVoice}
                    disabled={
                      analyzing ||
                      !registered
                    }
                    style={{
                      width: '100%',
                      marginTop: '16px',
                      justifyContent: 'center'
                    }}
                  >

                    <ShieldCheck size={18} />

                    {analyzing
                      ? 'Analyzing Voice…'
                      : registered
                        ? 'Analyze Voice'
                        : 'Enroll Reference First'}

                  </button>

                )}


              {error && (

                <div
                  style={{
                    marginTop: '15px',
                    padding: '12px',
                    borderRadius: '10px',
                    background: '#fff1f1',
                    color: '#b42318'
                  }}
                >
                  {error}
                </div>

              )}

            </div>


            {/* STEP 03 */}

            <div className="card analysis-card">

              <div className="card-head">

                <div>

                  <span className="kicker">
                    STEP 03
                  </span>

                  <h2>
                    AI Analysis
                  </h2>

                </div>

                <ShieldCheck size={22} />

              </div>


              {!result ? (

                <div className="coming">

                  <div className="scan">
                    <div className="scan-line"></div>
                    <Activity size={38} />
                  </div>

                  <h3>
                    {analyzing
                      ? 'Analyzing Voice…'
                      : 'Analysis Engine'}
                  </h3>

                  <p>
                    VoxShield combines AI deepfake
                    detection with speaker verification
                    to assess voice authenticity.
                  </p>


                  <div className="pipeline">

                    <span>Audio</span>
                    <i>→</i>
                    <span>Deepfake AI</span>
                    <i>→</i>
                    <span>ECAPA</span>
                    <i>→</i>
                    <span>Risk Score</span>

                  </div>


                  <div className="future">

                    <CheckCircle2 size={15} />

                    AI detection and speaker
                    verification connected

                  </div>

                </div>

              ) : (

                <div className="security-assessment">

                  <div className="assessment-header">

                    <div className="assessment-icon">

                      {result.risk_level === 'HIGH'
                        ? <AlertTriangle size={28} />
                        : <ShieldCheck size={28} />}

                    </div>


                    <div>

                      <span className="kicker">
                        SECURITY ASSESSMENT
                      </span>

                      <h3>
                        {result.decision ===
                        'AUTHENTIC_VOICE'
                          ? 'Voice Appears Authentic'
                          : result.decision ===
                            'POSSIBLE_VOICE_CLONING'
                            ? 'Possible Voice Cloning'
                            : result.decision ===
                              'SPOOFED_VOICE_DETECTED'
                              ? 'Possible AI-Generated Voice'
                              : 'Voice Security Alert'}
                      </h3>

                      <p>
                        {result.risk_level === 'LOW'
                          ? 'Interaction can proceed normally.'
                          : 'Additional verification is recommended before proceeding.'}
                      </p>

                    </div>

                  </div>


                  <div className="score-card">

                    <span>
                      Authenticity Score
                    </span>

                    <strong>
                      {result.authenticity_score}%
                    </strong>

                  </div>


                  <div className="result-grid">

                    <div className="result-item">
                      <span>
                        AI / Spoof Probability
                      </span>

                      <strong>
                        {(
                          result.spoof_probability *
                          100
                        ).toFixed(2)}%
                      </strong>
                    </div>


                    <div className="result-item">
                      <span>
                        Deepfake Detection
                      </span>

                      <strong>
                        {result.prediction ||
                          'N/A'}
                      </strong>
                    </div>


                    <div className="result-item">
                      <span>
                        Speaker Similarity
                      </span>

                      <strong>
                        {(
                          result.speaker_similarity *
                          100
                        ).toFixed(2)}%
                      </strong>
                    </div>


                    <div className="result-item">
                      <span>
                        Speaker Match
                      </span>

                      <strong>
                        {result.speaker_match
                          ? 'YES'
                          : 'NO'}
                      </strong>
                    </div>


                    <div className="result-item">
                      <span>
                        Risk Level
                      </span>

                      <strong>
                        {result.risk_level}
                      </strong>
                    </div>

                  </div>


                  <div className="decision-card">

                    <span>
                      Decision
                    </span>

                    <strong>
                      {result.decision}
                    </strong>

                  </div>


                  <div className="interaction-status">

                    {result.risk_level === 'LOW'
                      ? <CheckCircle2 size={18} />
                      : <AlertTriangle size={18} />}

                    <span>

                      <b>
                        Interaction{' '}
                        {result.risk_level === 'LOW'
                          ? 'Allowed'
                          : 'Requires Verification'}
                      </b>

                      <small>
                        {result.risk_level === 'LOW'
                          ? 'Voice passed the current security assessment.'
                          : 'Do not rely on voice identity alone.'}
                      </small>

                    </span>

                  </div>

                </div>

              )}

            </div>

          </section>


          {/* WHY VOXSHIELD */}

          <section className="why-voxshield">

            <div className="why-header">

              <span className="section-eyebrow">
                WHY VOXSHIELD
              </span>

              <h2>
                Built for safer voice interactions
              </h2>

              <p>
                Multiple layers of protection work
                together to detect suspicious voice activity.
              </p>

            </div>


            <div className="protection-grid">

              <div className="protection-card">

                <div className="protection-icon">
                  ✓
                </div>

                <div>

                  <h3>
                    Real-time detection
                  </h3>

                  <p>
                    Low-latency AI analysis of
                    incoming voice recordings.
                  </p>

                </div>

              </div>


              <div className="protection-card">

                <div className="protection-icon">
                  ♙
                </div>

                <div>

                  <h3>
                    Speaker verification
                  </h3>

                  <p>
                    Incoming voices are checked
                    against the enrolled speaker profile.
                  </p>

                </div>

              </div>


              <div className="protection-card">

                <div className="protection-icon">
                  ⚠
                </div>

                <div>

                  <h3>
                    Threat prevention
                  </h3>

                  <p>
                    Suspicious voice activity is
                    identified before access is allowed.
                  </p>

                </div>

              </div>

            </div>


            <div className="protection-footer">

              <span>AASIST</span>
              <span>•</span>
              <span>ECAPA-TDNN</span>
              <span>•</span>
              <span>Risk Engine</span>

            </div>

          </section>

        </>
      )}


      {/* ================================
          DASHBOARD PAGE
          ================================ */}

      {activePage === 'dashboard' && (

        <section className="dashboard">

          <div className="dashboard-head">

            <div>

              <div className="eyebrow">
                SECURITY DASHBOARD
              </div>

              <h1>
                Voice protection,{' '}
                <em>at a glance.</em>
              </h1>

              <p>
                Monitor your enrolled identity
                and recent voice security assessments.
              </p>

            </div>


            <button
              className="primary dashboard-cta"
              onClick={() =>
                setActivePage('analysis')
              }
            >
              <Mic size={17} />
              Start Voice Analysis
            </button>

          </div>


          {/* STAT CARDS */}

          <div className="stat-grid">

            <div className="stat-card">

              <span>
                PROTECTION STATUS
              </span>

              <strong
                className={
                  registered
                    ? 'ok'
                    : 'warn'
                }
              >
                {registered
                  ? 'ACTIVE'
                  : 'SETUP REQUIRED'}
              </strong>

              <small>
                {registered
                  ? `${referenceCount} reference recording${
                      referenceCount === 1
                        ? ''
                        : 's'
                    } • ECAPA profile ready`
                  : 'Enroll a reference voice'}
              </small>

            </div>


            <div className="stat-card">

              <span>
                ANALYSES THIS SESSION
              </span>

              <strong>
                {analysisHistory.length}
              </strong>

              <small>
                Voice samples assessed
              </small>

            </div>


            <div className="stat-card">

              <span>
                THREATS DETECTED
              </span>

              <strong>
                {
                  analysisHistory.filter(
                    item =>
                      item.result?.risk_level ===
                      'HIGH'
                  ).length
                }
              </strong>

              <small>
                High-risk assessments
              </small>

            </div>


            <div className="stat-card">

              <span>
                ALLOWED
              </span>

              <strong>
                {
                  analysisHistory.filter(
                    item =>
                      item.result?.risk_level ===
                      'LOW'
                  ).length
                }
              </strong>

              <small>
                Low-risk assessments
              </small>

            </div>

          </div>


          {/* DASHBOARD CONTENT */}

          <div className="dashboard-grid">

            {/* LATEST SECURITY ASSESSMENT */}

            <div className="card latest-card">

              <div className="card-head">

                <div>

                  <span className="kicker">
                    LATEST RESULT
                  </span>

                  <h2>
                    {analysisHistory.length
                      ? 'Latest Security Assessment'
                      : 'No Analysis Yet'}
                  </h2>

                </div>


                {analysisHistory.length ? (

                  analysisHistory[0].result
                    ?.risk_level === 'HIGH'
                    ? <AlertTriangle size={22} />
                    : <ShieldCheck size={22} />

                ) : (

                  <Activity size={22} />

                )}

              </div>


              {analysisHistory.length ? (

                (() => {

                  const latest =
                    analysisHistory[0].result;

                  return (

                    <>

                      <div
                        className={`risk-banner ${
                          latest.risk_level
                            ?.toLowerCase()
                        }`}
                      >

                        <div>

                          <small>
                            RISK LEVEL
                          </small>

                          <b>
                            {latest.risk_level}
                          </b>

                        </div>


                        <div>

                          <small>
                            AUTHENTICITY
                          </small>

                          <b>
                            {latest.authenticity_score ??
                              '--'}
                            %
                          </b>

                        </div>

                      </div>


                      <div className="dashboard-metrics">

                        <div>

                          <span>
                            AI / SPOOF
                          </span>

                          <b>
                            {(
                              (latest.spoof_probability ??
                                0) *
                              100
                            ).toFixed(2)}
                            %
                          </b>

                        </div>


                        <div>

                          <span>
                            SPEAKER SIMILARITY
                          </span>

                          <b>
                            {(
                              (latest.speaker_similarity ??
                                0) *
                              100
                            ).toFixed(2)}
                            %
                          </b>

                        </div>


                        <div>

                          <span>
                            SPEAKER MATCH
                          </span>

                          <b>
                            {latest.speaker_match
                              ? 'YES'
                              : 'NO'}
                          </b>

                        </div>

                      </div>


                      <div className="decision-card">

                        <span>
                          DECISION
                        </span>

                        <strong>
                          {latest.decision}
                        </strong>

                      </div>

                    </>

                  );

                })()

              ) : (

                <div className="empty-dashboard">

                  <Activity size={30} />

                  <h3>
                    Ready for your first scan
                  </h3>

                  <p>
                    Run a voice analysis and your
                    latest security assessment
                    will appear here.
                  </p>


                  <button
                    className="secondary"
                    style={{
                      marginTop: '18px'
                    }}
                    onClick={() =>
                      setActivePage('analysis')
                    }
                  >
                    <Mic size={16} />
                    Open Analysis
                  </button>

                </div>

              )}

            </div>


            {/* PROTECTION PIPELINE */}

            <div className="card system-card">

              <div className="card-head">

                <div>

                  <span className="kicker">
                    ENGINE STATUS
                  </span>

                  <h2>
                    Protection Pipeline
                  </h2>

                </div>

                <LockKeyhole size={22} />

              </div>


              <div className="engine-row">

                <span className="engine-icon">
                  <CheckCircle2 size={16} />
                </span>

                <div>

                  <b>
                    AASIST Anti-Spoofing
                  </b>

                  <small>
                    AI / synthetic voice detection
                  </small>

                </div>

                <strong>
                  READY
                </strong>

              </div>


              <div className="engine-row">

                <span className="engine-icon">
                  <CheckCircle2 size={16} />
                </span>

                <div>

                  <b>
                    ECAPA-TDNN
                  </b>

                  <small>
                    Speaker identity verification
                  </small>

                </div>

                <strong>
                  READY
                </strong>

              </div>


              <div className="engine-row">

                <span className="engine-icon">
                  <CheckCircle2 size={16} />
                </span>

                <div>

                  <b>
                    Risk Engine
                  </b>

                  <small>
                    Allow / Verify / Block decision
                  </small>

                </div>

                <strong>
                  READY
                </strong>

              </div>


              <button
                className="secondary full-btn"
                onClick={() =>
                  setActivePage('analysis')
                }
              >
                <Activity size={16} />
                Open Analysis Console
              </button>

            </div>

          </div>


          {/* SESSION SUMMARY */}

          <div className="session-strip">

            <div>

              <span>
                SESSION ACTIVITY
              </span>

              <b>
                {analysisHistory.length}
              </b>

              <small>
                Total analyses
              </small>

            </div>


            <div>

              <span>
                VERIFICATION
              </span>

              <b>
                {
                  analysisHistory.filter(
                    item =>
                      item.result?.risk_level ===
                      'MEDIUM'
                  ).length
                }
              </b>

              <small>
                Requires verification
              </small>

            </div>


            <div>

              <span>
                THREATS
              </span>

              <b>
                {
                  analysisHistory.filter(
                    item =>
                      item.result?.risk_level ===
                      'HIGH'
                  ).length
                }
              </b>

              <small>
                High-risk detections
              </small>

            </div>


            <div>

              <span>
                ENGINE
              </span>

              <b>
                ONLINE
              </b>

              <small>
                AASIST + ECAPA + Risk
              </small>

            </div>

          </div>

        </section>

      )}


      {/* ================================
          HISTORY PAGE
          ================================ */}

      {activePage === 'history' && (

        <section className="history-page">

          <div className="history-hero">

            <div>

              <span className="section-eyebrow">
                SECURITY HISTORY
              </span>

              <h1>
                Analysis History
              </h1>

              <p>
                Detection results, confidence scores,
                and security assessments from the
                current session.
              </p>

            </div>


            <div className="history-count">

              <strong>
                {analysisHistory.length}
              </strong>

              <span>
                Analyses
              </span>

            </div>

          </div>


          {analysisHistory.length === 0 ? (

            <div className="history-empty">

              <div className="history-empty-icon">
                <Activity size={25} />
              </div>

              <h2>
                No analysis history yet
              </h2>

              <p>
                Completed voice analyses will appear
                here with their detection and
                verification results.
              </p>

              <button
                className="history-action"
                onClick={() =>
                  setActivePage('analysis')
                }
              >
                <Mic size={15} />
                Start Voice Analysis
              </button>

            </div>

          ) : (

            <div className="history-list">

              {analysisHistory.map(item => {

                const r = item.result;

                const riskClass =
                  r.risk_level === 'HIGH'
                    ? 'risk-high'
                    : r.risk_level === 'MEDIUM'
                      ? 'risk-medium'
                      : 'risk-low';


                const decisionTitle =
                  r.decision ===
                  'AUTHENTIC_VOICE'
                    ? 'Authentic Voice'
                    : r.decision ===
                      'POSSIBLE_VOICE_CLONING'
                      ? 'Possible Voice Cloning'
                      : r.decision ===
                        'SPOOFED_VOICE_DETECTED'
                        ? 'Spoofed Voice Detected'
                        : 'Possible Speaker Mismatch';


                return (

                  <article
                    className="history-card"
                    key={item.id}
                  >

                    <div className="history-card-top">

                      <div>

                        <span className="history-time">
                          {item.time}
                        </span>

                        <h2>
                          {decisionTitle}
                        </h2>

                      </div>


                      <span
                        className={`history-risk ${riskClass}`}
                      >
                        {r.risk_level} RISK
                      </span>

                    </div>


                    <div className="history-divider" />


                    <div className="history-metrics">

                      <div className="history-metric">

                        <span>
                          SPOOF PROBABILITY
                        </span>

                        <strong>
                          {(
                            (r.spoof_probability ??
                              0) *
                            100
                          ).toFixed(2)}
                          %
                        </strong>

                      </div>


                      <div className="history-metric">

                        <span>
                          SPEAKER SIMILARITY
                        </span>

                        <strong>
                          {(
                            (r.speaker_similarity ??
                              0) *
                            100
                          ).toFixed(2)}
                          %
                        </strong>

                      </div>


                      <div className="history-metric">

                        <span>
                          SPEAKER MATCH
                        </span>

                        <strong>
                          {r.speaker_match
                            ? 'YES'
                            : 'NO'}
                        </strong>

                      </div>


                      <div className="history-metric">

                        <span>
                          AUTHENTICITY
                        </span>

                        <strong>
                          {Number(
                            r.authenticity_score ??
                              0
                          ).toFixed(2)}
                          %
                        </strong>

                      </div>

                    </div>


                    <div className="history-evidence">

                      <span>
                        AASIST
                      </span>

                      <b>
                        {r.prediction === 'SPOOF'
                          ? 'SPOOF DETECTED'
                          : 'BONAFIDE'}
                      </b>


                      <span>
                        ECAPA-TDNN
                      </span>

                      <b>
                        {r.speaker_match
                          ? 'SPEAKER MATCHED'
                          : 'MISMATCH'}
                      </b>


                      <span>
                        RISK ENGINE
                      </span>

                      <b>
                        {r.decision}
                      </b>

                    </div>

                  </article>

                );

              })}

            </div>

          )}

        </section>

      )}


      {/* FOOTER */}

      <footer>
        VoxShield • SIH 2026 • Will Improve More In Future
      </footer>

    </div>
  );
}

createRoot(
  document.getElementById('root')
).render(<App />);