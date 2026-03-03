import { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

/* ─────────────────────────────────────────────────────────────
   CONFIG
   ───────────────────────────────────────────────────────────── */
const WS_URL = "ws://localhost:8000/ws/predict";
const NUM_HANDS = 2;
const NUM_LANDMARKS = 21;
const FEATURES_PER_HAND = NUM_LANDMARKS * 3; // 63
const TOTAL_FEATURES = NUM_HANDS * FEATURES_PER_HAND; // 126

/* ─────────────────────────────────────────────────────────────
   HAND SKELETON CONNECTIONS (for drawing)
   ───────────────────────────────────────────────────────────── */
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

function App() {
  /* ── refs ──────────────────────────────────────────────── */
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const wsRef = useRef(null);
  const animFrameRef = useRef(null);
  const lastTimestampRef = useRef(-1);
  const isRecordingRef = useRef(false); // mirror of state for rAF loop

  /* ── state ─────────────────────────────────────────────── */
  const [gesture, setGesture] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [wsStatus, setWsStatus] = useState("disconnected");
  const [mpReady, setMpReady] = useState(false);
  const [buffering, setBuffering] = useState(0);
  const [isRecording, setIsRecording] = useState(false);

  /* ─────────────────────────────────────────────────────────
     TOGGLE RECORDING
     ───────────────────────────────────────────────────────── */
  const startRecording = useCallback(() => {
    // Clear previous result & buffer
    setGesture("");
    setConfidence(0);
    setBuffering(0);
    // Tell backend to clear its deque too
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "clear" }));
    }
    isRecordingRef.current = true;
    setIsRecording(true);
  }, []);

  const stopRecording = useCallback(() => {
    isRecordingRef.current = false;
    setIsRecording(false);
    setBuffering(0);
  }, []);

  /* ─────────────────────────────────────────────────────────
     1. INIT MEDIAPIPE HAND LANDMARKER
     ───────────────────────────────────────────────────────── */
  useEffect(() => {
    let cancelled = false;

    async function initMediaPipe() {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      const handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: NUM_HANDS,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      if (!cancelled) {
        landmarkerRef.current = handLandmarker;
        setMpReady(true);
      }
    }

    initMediaPipe();
    return () => { cancelled = true; };
  }, []);

  /* ─────────────────────────────────────────────────────────
     2. WEBSOCKET CONNECTION
     ───────────────────────────────────────────────────────── */
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setWsStatus("connecting");

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => setWsStatus("connected");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === "prediction") {
        setGesture(data.gesture);
        setConfidence(data.confidence);
        setBuffering(0);
        // Auto-stop after successful prediction
        isRecordingRef.current = false;
        setIsRecording(false);
      } else if (data.status === "low_confidence") {
        setGesture("");
        setConfidence(data.confidence);
        setBuffering(0);
        // Auto-stop — user can retry
        isRecordingRef.current = false;
        setIsRecording(false);
      } else if (data.status === "buffering") {
        setBuffering(data.frames_collected);
      }
    };

    ws.onclose = () => {
      setWsStatus("disconnected");
      setTimeout(connectWs, 2000);
    };

    ws.onerror = () => ws.close();

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connectWs();
    return () => wsRef.current?.close();
  }, [connectWs]);

  /* ─────────────────────────────────────────────────────────
     3. MAIN DETECTION + DRAW + SEND LOOP
     ───────────────────────────────────────────────────────── */
  const drawLandmarks = useCallback((results, ctx, w, h) => {
    ctx.clearRect(0, 0, w, h);
    if (!results?.landmarks?.length) return;

    for (const hand of results.landmarks) {
      ctx.strokeStyle = isRecordingRef.current
        ? "rgba(0,200,255,0.8)"
        : "rgba(255,255,255,0.6)";
      ctx.lineWidth = 2;
      for (const [start, end] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(hand[start].x * w, hand[start].y * h);
        ctx.lineTo(hand[end].x * w, hand[end].y * h);
        ctx.stroke();
      }
      for (const lm of hand) {
        ctx.beginPath();
        ctx.arc(lm.x * w, lm.y * h, 4, 0, 2 * Math.PI);
        ctx.fillStyle = isRecordingRef.current ? "#00d4ff" : "#00ff80";
        ctx.fill();
      }
    }
  }, []);

  const extractAndSend = useCallback((results) => {
    // Only send when recording
    if (!isRecordingRef.current) return;
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;

    const left = new Array(FEATURES_PER_HAND).fill(0);
    const right = new Array(FEATURES_PER_HAND).fill(0);

    if (results?.landmarks?.length) {
      results.landmarks.forEach((hand, i) => {
        const handedness = results.handednesses?.[i]?.[0]?.categoryName;
        const coords = hand.flatMap((lm) => [lm.x, lm.y, lm.z]);
        if (handedness === "Left") {
          coords.forEach((v, j) => (left[j] = v));
        } else {
          coords.forEach((v, j) => (right[j] = v));
        }
      });
    }

    wsRef.current.send(JSON.stringify({ landmarks: [...left, ...right] }));
  }, []);

  const detect = useCallback(() => {
    const video = webcamRef.current?.video;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;

    if (!video || !canvas || !landmarker || video.readyState < 2) {
      animFrameRef.current = requestAnimationFrame(detect);
      return;
    }

    const w = video.videoWidth;
    const h = video.videoHeight;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");

    const now = performance.now();
    if (now <= lastTimestampRef.current) {
      animFrameRef.current = requestAnimationFrame(detect);
      return;
    }
    lastTimestampRef.current = now;

    const results = landmarker.detectForVideo(video, now);

    drawLandmarks(results, ctx, w, h);
    extractAndSend(results);

    animFrameRef.current = requestAnimationFrame(detect);
  }, [drawLandmarks, extractAndSend]);

  useEffect(() => {
    if (mpReady) {
      animFrameRef.current = requestAnimationFrame(detect);
    }
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [mpReady, detect]);

  /* ─────────────────────────────────────────────────────────
     4. RENDER
     ───────────────────────────────────────────────────────── */
  const statusColor =
    wsStatus === "connected"
      ? "bg-emerald-500"
      : wsStatus === "connecting"
        ? "bg-yellow-500"
        : "bg-red-500";

  const canStart = mpReady && wsStatus === "connected" && !isRecording;

  return (
    <div className="relative min-h-screen w-full bg-gray-950 overflow-hidden flex flex-col items-center justify-center gap-6 p-4">
      {/* ── Webcam container ──────────────────────────────── */}
      <div className="relative w-full max-w-4xl aspect-video rounded-2xl overflow-hidden shadow-2xl shadow-black/60 border border-white/10">
        <Webcam
          ref={webcamRef}
          mirrored
          className="absolute inset-0 w-full h-full object-cover"
          videoConstraints={{ facingMode: "user", width: 1280, height: 720 }}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* ── Recording border glow ─────────────────────── */}
        {isRecording && (
          <div className="absolute inset-0 rounded-2xl border-2 border-cyan-400/60 pointer-events-none animate-pulse" />
        )}

        {/* ── Top status bar ────────────────────────────── */}
        <div className="absolute top-0 inset-x-0 flex items-center justify-between px-5 py-3 bg-gradient-to-b from-black/70 to-transparent">
          <div className="flex items-center gap-2">
            <span className="text-white/90 font-bold text-lg tracking-wide">
              🧠 Neuro-Sign
            </span>
            {isRecording && (
              <span className="flex items-center gap-1.5 ml-3 text-xs font-semibold text-cyan-400 animate-pulse">
                <span className="inline-block w-2 h-2 rounded-full bg-red-500" />
                Recording…
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-white/60">
              {mpReady ? "✔ MediaPipe" : "⏳ Loading model…"}
            </span>
            <div className="flex items-center gap-1.5">
              <span className={`inline-block w-2 h-2 rounded-full ${statusColor} animate-pulse`} />
              <span className="text-xs text-white/60 capitalize">{wsStatus}</span>
            </div>
          </div>
        </div>

        {/* ── Buffering progress ────────────────────────── */}
        {isRecording && buffering > 0 && buffering < 30 && (
          <div className="absolute top-14 inset-x-0 flex justify-center">
            <div className="bg-black/60 backdrop-blur-sm rounded-full px-4 py-1.5">
              <div className="flex items-center gap-2">
                <div className="w-32 h-1.5 bg-white/20 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-cyan-400 rounded-full transition-all duration-100"
                    style={{ width: `${(buffering / 30) * 100}%` }}
                  />
                </div>
                <span className="text-xs text-white/50">{buffering}/30</span>
              </div>
            </div>
          </div>
        )}

        {/* ── Gesture subtitle overlay ───────────────────── */}
        <div className="absolute bottom-0 inset-x-0 flex justify-center pb-8 pointer-events-none">
          {gesture ? (
            <div className="bg-black/70 backdrop-blur-md border border-white/10 rounded-2xl px-8 py-4 shadow-lg shadow-black/40 animate-fade-in">
              <p className="text-3xl font-bold text-white text-center tracking-wide">
                {gesture}
              </p>
              <div className="mt-1 flex justify-center">
                <span className="text-xs text-emerald-400 font-medium">
                  {(confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
            </div>
          ) : (
            <div className="bg-black/40 backdrop-blur-sm rounded-2xl px-6 py-3">
              <p className="text-sm text-white/40 text-center">
                {!mpReady
                  ? "Loading hand detection model…"
                  : isRecording
                    ? "Recording gesture — hold your pose…"
                    : "Press the button below to start"}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── Start / Stop button ───────────────────────────── */}
      <button
        onClick={isRecording ? stopRecording : startRecording}
        disabled={!canStart && !isRecording}
        className={`
          relative group px-10 py-4 rounded-2xl font-bold text-lg tracking-wide
          transition-all duration-300 ease-out
          disabled:opacity-40 disabled:cursor-not-allowed
          ${
            isRecording
              ? "bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-600/30"
              : "bg-cyan-500 hover:bg-cyan-400 text-gray-950 shadow-lg shadow-cyan-500/30 hover:shadow-cyan-400/40 hover:scale-105"
          }
        `}
      >
        {/* Animated ring behind the button when recording */}
        {isRecording && (
          <span className="absolute inset-0 rounded-2xl border-2 border-red-400/50 animate-ping pointer-events-none" />
        )}

        {isRecording ? "⏹  Stop Gesture" : "🤟  Start Gesture"}
      </button>

      {/* ── Hint text ─────────────────────────────────────── */}
      <p className="text-xs text-white/30 text-center max-w-md">
        {!mpReady
          ? "Initializing hand detection model — this may take a moment…"
          : isRecording
            ? "Hold your gesture steady. Recording 30 frames for prediction."
            : gesture
              ? "Prediction complete! Click Start Gesture to try again."
              : "Position your hand in view, then click Start Gesture."}
      </p>
    </div>
  );
}

export default App;
