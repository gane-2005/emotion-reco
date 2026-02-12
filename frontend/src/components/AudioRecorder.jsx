import { useState, useRef, useEffect } from 'react';
import { predictEmotion } from '../utils/api';
import { encodeWAV, flattenArray } from '../utils/WavEncoder';

function AudioRecorder({ onPredictionComplete, onLoading }) {
    const [isRecording, setIsRecording] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const [error, setError] = useState('');

    const contextRef = useRef(null);
    const processorRef = useRef(null);
    const sourceRef = useRef(null);
    const streamRef = useRef(null);

    // Use a useRef for the chunks because the processor callback runs outside React state context
    const audioChunksRef = useRef([]);
    const recordingLengthRef = useRef(0);

    const timerRef = useRef(null);

    // Cleanup function when component unmounts
    useEffect(() => {
        return () => {
            stopRecordingProcess();
        };
    }, []);

    const startRecording = async () => {
        try {
            setError('');

            // Get audio stream
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            // Initialize AudioContext
            contextRef.current = new (window.AudioContext || window.webkitAudioContext)();
            const context = contextRef.current;

            // Create source
            sourceRef.current = context.createMediaStreamSource(stream);
            const source = sourceRef.current;

            // Create processor (bufferSize, inputChannels, outputChannels)
            // 4096 is a good balance for latency/performance
            processorRef.current = context.createScriptProcessor(4096, 1, 1);
            const processor = processorRef.current;


            // Reset buffers
            audioChunksRef.current = [];
            recordingLengthRef.current = 0;

            // Handle audio processing
            processor.onaudioprocess = (e) => {
                if (!isRecording && audioChunksRef.current.length === 0 && !streamRef.current) return; // Guard clause

                const inputData = e.inputBuffer.getChannelData(0);

                // Clone the data (important because buffers are reused)
                const dataCopy = new Float32Array(inputData);
                audioChunksRef.current.push(dataCopy);
                recordingLengthRef.current += dataCopy.length;
            };

            // Connect nodes: source -> processor -> destination
            source.connect(processor);
            processor.connect(context.destination);

            setIsRecording(true);
            setRecordingTime(0);

            // Start timer
            timerRef.current = setInterval(() => {
                setRecordingTime(prev => prev + 1);
            }, 1000);

        } catch (err) {
            console.error('Error accessing microphone:', err);
            setError('Failed to access microphone. Please grant permission.');
        }
    };

    const stopRecordingHelper = async () => {
        // Stop gathering data
        setIsRecording(false);
        if (timerRef.current) clearInterval(timerRef.current);

        // Process audio data
        if (audioChunksRef.current.length > 0) {
            const sampleRate = contextRef.current.sampleRate;

            // Flatten and Encode
            const flatData = flattenArray(audioChunksRef.current, recordingLengthRef.current);
            const wavData = encodeWAV(flatData, sampleRate);

            const blob = new Blob([wavData], { type: 'audio/wav' });
            const file = new File([blob], `recording-${Date.now()}.wav`, { type: 'audio/wav' });

            // Upload
            await handleUpload(file);
        }

        // Cleanup resources
        stopRecordingProcess();
    };

    const stopRecordingProcess = () => {
        if (processorRef.current && sourceRef.current) {
            sourceRef.current.disconnect();
            processorRef.current.disconnect();
        }

        if (contextRef.current && contextRef.current.state !== 'closed') {
            contextRef.current.close();
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        // Clear refs
        processorRef.current = null;
        sourceRef.current = null;
        contextRef.current = null;
    };

    const handleUpload = async (file) => {
        try {
            onLoading(true);
            const result = await predictEmotion(file);

            if (result.success) {
                onPredictionComplete(result);
            } else {
                setError(result.error || 'Prediction failed');
            }
        } catch (err) {
            console.error('Prediction error:', err);
            setError(err.error || 'Failed to analyze recording. Please ensure the backend is running.');
        } finally {
            onLoading(false);
        }
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="audio-recorder">
            <div className="recorder-container">
                {!isRecording ? (
                    <button
                        className="record-button start"
                        onClick={startRecording}
                        title="Start Recording"
                    >
                        <span className="record-icon">🎙️</span>
                        <span className="record-text">Start Recording</span>
                    </button>
                ) : (
                    <div className="recording-active">
                        <div className="recording-indicator">
                            <span className="pulse"></span>
                            <span className="recording-text">Recording...</span>
                        </div>
                        <div className="recording-time">{formatTime(recordingTime)}</div>
                        <button
                            className="record-button stop"
                            onClick={stopRecordingHelper}
                            title="Stop Recording"
                        >
                            <span className="stop-icon">⏹️</span>
                            <span className="stop-text">Stop & Analyze</span>
                        </button>
                    </div>
                )}
            </div>

            {error && (
                <div className="error-message">
                    <span className="error-icon">❌</span> {error}
                </div>
            )}

            <p className="recorder-hint">
                💡 Speak clearly for 3-5 seconds for best results
            </p>
        </div>
    );
}

export default AudioRecorder;
