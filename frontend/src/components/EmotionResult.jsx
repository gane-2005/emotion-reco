function EmotionResult({ result, loading }) {
    if (loading) {
        return (
            <div className="emotion-result loading">
                <div className="loader"></div>
                <p>Analyzing emotion...</p>
            </div>
        );
    }

    if (!result) {
        return null;
    }

    // Emotion emoji mapping
    const emotionEmojis = {
        'Happy': '😊',
        'Sad': '😢',
        'Angry': '😠',
        'Fear': '😨',
        'Surprise': '😲',
        'Disgust': '🤢',
        'Neutral': '😐'
    };

    // Emotion color mapping
    const emotionColors = {
        'Happy': '#FFD700',
        'Sad': '#4A90E2',
        'Angry': '#E74C3C',
        'Fear': '#9B59B6',
        'Surprise': '#F39C12',
        'Disgust': '#16A085',
        'Neutral': '#95A5A6'
    };

    const { emotion, confidence, all_probabilities } = result;
    const emoji = emotionEmojis[emotion] || '🎭';
    const color = emotionColors[emotion] || '#3498DB';

    return (
        <div className="emotion-result" style={{ borderColor: color }}>
            {/* Main Result */}
            <div className="main-result" style={{ backgroundColor: `${color}15` }}>
                <div className="emotion-emoji" style={{ color }}>
                    {emoji}
                </div>
                <h2 className="emotion-label" style={{ color }}>
                    {emotion}
                </h2>
                <div className="confidence-container">
                    <div className="confidence-label">Confidence</div>
                    <div className="confidence-value">{confidence.toFixed(1)}%</div>
                    <div className="progress-bar">
                        <div
                            className="progress-fill"
                            style={{
                                width: `${confidence}%`,
                                backgroundColor: color
                            }}
                        ></div>
                    </div>
                </div>
            </div>

            {/* All Probabilities */}
            {all_probabilities && (
                <div className="all-probabilities">
                    <h3>📊 Detailed Analysis</h3>
                    <div className="probability-grid">
                        {Object.entries(all_probabilities)
                            .sort(([, a], [, b]) => b - a)
                            .map(([em, prob]) => (
                                <div key={em} className="probability-item">
                                    <div className="prob-header">
                                        <span className="prob-emoji">{emotionEmojis[em]}</span>
                                        <span className="prob-label">{em}</span>
                                    </div>
                                    <div className="prob-bar">
                                        <div
                                            className="prob-fill"
                                            style={{
                                                width: `${prob}%`,
                                                backgroundColor: emotionColors[em]
                                            }}
                                        ></div>
                                    </div>
                                    <span className="prob-value">{prob.toFixed(1)}%</span>
                                </div>
                            ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default EmotionResult;
