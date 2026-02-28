import { useEffect, useState } from 'react';
import { getStats } from '../utils/api';

const EMOTION_EMOJIS = {
    Happy: '😊', Sad: '😢', Angry: '😠',
    Fear: '😨', Surprise: '😲', Disgust: '🤢', Neutral: '😐'
};

const EMOTION_COLORS = {
    Happy: '#FFD700', Sad: '#4A90E2', Angry: '#E74C3C',
    Fear: '#9B59B6', Surprise: '#F39C12', Disgust: '#16A085', Neutral: '#95A5A6'
};

function MoodDashboard() {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        fetchStats();
    }, []);

    const fetchStats = async () => {
        try {
            setLoading(true);
            const data = await getStats();
            if (data.success) setStats(data);
            else setError('Failed to load stats');
        } catch {
            setError('Could not connect to server');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="mood-dashboard loading">
                <div className="loader"></div>
                <p>Loading dashboard...</p>
            </div>
        );
    }

    if (error || !stats) {
        return (
            <div className="mood-dashboard error">
                <p>⚠️ {error || 'No data available'}</p>
            </div>
        );
    }

    const maxCount = stats.distribution.length > 0
        ? Math.max(...stats.distribution.map(d => d.count))
        : 1;

    return (
        <div className="mood-dashboard">
            {/* Summary Cards */}
            <div className="stats-summary-grid">
                <div className="stat-card">
                    <div className="stat-icon">🔢</div>
                    <div className="stat-number">{stats.total}</div>
                    <div className="stat-label">Total Analyses</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">📅</div>
                    <div className="stat-number">{stats.today}</div>
                    <div className="stat-label">Today</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">📆</div>
                    <div className="stat-number">{stats.this_week}</div>
                    <div className="stat-label">This Week</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">
                        {EMOTION_EMOJIS[stats.most_common_emotion] || '🎭'}
                    </div>
                    <div className="stat-number" style={{
                        color: EMOTION_COLORS[stats.most_common_emotion] || '#667eea',
                        fontSize: '1.2rem'
                    }}>
                        {stats.most_common_emotion || '—'}
                    </div>
                    <div className="stat-label">Most Common</div>
                </div>
            </div>

            {/* Emotion Distribution Chart */}
            {stats.distribution.length > 0 ? (
                <div className="distribution-chart">
                    <h4>Emotion Distribution</h4>
                    <div className="dist-bars">
                        {stats.distribution.map(({ emotion, count, avg_confidence }) => {
                            const color = EMOTION_COLORS[emotion] || '#667eea';
                            const pct = Math.round((count / maxCount) * 100);
                            return (
                                <div key={emotion} className="dist-row">
                                    <div className="dist-label">
                                        <span>{EMOTION_EMOJIS[emotion]} {emotion}</span>
                                        <span className="dist-meta">{count}x · {avg_confidence}%</span>
                                    </div>
                                    <div className="dist-bar-bg">
                                        <div
                                            className="dist-bar-fill"
                                            style={{ width: `${pct}%`, backgroundColor: color }}
                                        ></div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            ) : (
                <div className="no-data-message">
                    <p>🎤 No data yet — start analyzing voices to see your mood trends!</p>
                </div>
            )}
        </div>
    );
}

export default MoodDashboard;
