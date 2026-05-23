import React, { useState, useRef, useEffect } from 'react';
import './App.css';

// SVG Icons as inline components for clean packaging
const BottleIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 3h6v3H9z" />
    <path d="M5 9c0-1.7 1.3-3 3-3h8c1.7 0 3 1.3 3 3v10c0 1.7-1.3 3-3 3H8c-1.7 0-3-1.3-3-3V9z" />
    <path d="M12 9v10" />
    <path d="M9 13h6" />
  </svg>
);

const SparklesIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
    <path d="m5 3 1 2.5L8.5 6 6 7 5 9.5 4 7 1.5 6 4 5 5 3Z" />
    <path d="m19 17 1 2.5 2.5.5-2.5 1-1 2.5-1-2.5-2.5-1 2.5-1 1-2.5Z" />
  </svg>
);

const SendIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const SommelierAvatarIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z" />
    <path d="M12 14a3 3 0 1 0 0-6 3 3 0 0 0 0 6z" />
    <path d="M12 14v4" />
    <path d="M9 18h6" />
  </svg>
);

const UserAvatarIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
);

export default function App() {
  const [messages, setMessages] = useState([
    {
      sender: 'sommelier',
      text: "Hello! I am your personal fragrance sommelier. Tell me about the vibes, scents, or memories you are looking to capture in a perfume (e.g. 'something cozy and warm with vanilla, but fresh like a morning rain'), and I will find your perfect match."
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [genderFilter, setGenderFilter] = useState('');
  const [loading, setLoading] = useState(false);
  const [matches, setMatches] = useState([]);
  
  const messagesEndRef = useRef(null);

  // Scroll to bottom when messages list updates
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (e) => {
    if (e) e.preventDefault();
    if (!inputText.trim() || loading) return;

    const userMessage = inputText;
    setInputText('');
    setMessages(prev => [...prev, { sender: 'user', text: userMessage }]);
    setLoading(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || '';
      const response = await fetch(`${apiUrl}/api/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          description: userMessage,
          gender: genderFilter || null
        })
      });

      if (!response.ok) {
        throw new Error("Failed to retrieve recommendations from the sommelier.");
      }

      const data = await response.json();
      
      setMessages(prev => [...prev, { sender: 'sommelier', text: data.recommendation }]);
      if (data.matches && data.matches.length > 0) {
        setMatches(data.matches);
      }
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { 
        sender: 'sommelier', 
        text: "My apologies. I encountered a slight disturbance in the air while sniffing out notes. Please make sure the backend is active and try again shortly!" 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleChipClick = (suggestion) => {
    setInputText(suggestion);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo-section">
          <span className="logo-icon"><BottleIcon /></span>
          <div>
            <h1>Fragrance Finder</h1>
            <span>Conversational Recommender</span>
          </div>
        </div>
        <div className="badge">
          Groq + pgvector
        </div>
      </header>

      {/* Main Workspace */}
      <div className="main-content">
        {/* Chat Area */}
        <section className="chat-area">
          <div className="messages-container">
            {messages.length === 1 && (
              <div className="welcome-card">
                <div className="welcome-icon-wrapper">
                  <SparklesIcon />
                </div>
                <h2>Discover Your Scent</h2>
                <p>
                  Describe the mood, notes, or seasons you want, and let the sommelier retrieve similar items using vector matching.
                </p>
                <div className="suggestion-chips">
                  <button onClick={() => handleChipClick("A fresh, marine fragrance with sea salt and sage for hot summer days")} className="chip">
                    🌊 Marine & Sage
                  </button>
                  <button onClick={() => handleChipClick("Cozy, sweet vanilla with a smoky tobacco undertone")} className="chip">
                    🍂 Smoky Vanilla
                  </button>
                  <button onClick={() => handleChipClick("Clean, soapy rose with powdery musk")} className="chip">
                    🌹 Powdery Rose
                  </button>
                </div>
              </div>
            )}

            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                <div className="avatar">
                  {msg.sender === 'sommelier' ? <SommelierAvatarIcon /> : <UserAvatarIcon />}
                </div>
                <div className="bubble">
                  {msg.text}
                </div>
              </div>
            ))}
            
            {loading && (
              <div className="message sommelier">
                <div className="avatar">
                  <SommelierAvatarIcon />
                </div>
                <div className="bubble loading-message">
                  <span>Sommelier is sniffing notes</span>
                  <div className="scent-wave">
                    <span className="scent-dot"></span>
                    <span className="scent-dot"></span>
                    <span className="scent-dot"></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Chat Input Bar */}
          <div className="chat-input-container">
            <form onSubmit={handleSend} className="input-form">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Describe your perfect fragrance profile..."
                className="text-input"
                disabled={loading}
              />
              <select
                value={genderFilter}
                onChange={(e) => setGenderFilter(e.target.value)}
                className="gender-select"
                disabled={loading}
              >
                <option value="">All Genders</option>
                <option value="unisex">Unisex</option>
                <option value="women">Female</option>
                <option value="men">Male</option>
              </select>
              <button type="submit" disabled={loading || !inputText.trim()} className="send-button">
                <SendIcon />
              </button>
            </form>
          </div>
        </section>

        {/* Sidebar Matches Panel */}
        <aside className="matches-panel">
          <h3>
            <BottleIcon /> Recommended Candidates
          </h3>
          {matches.length === 0 ? (
            <div className="empty-matches">
              <SparklesIcon />
              <p>No active matches yet.</p>
              <p style={{ fontSize: '0.8rem' }}>Once the sommelier makes recommendations, the detailed fragrance profiles will populate here.</p>
            </div>
          ) : (
            <div className="matches-list">
              {matches.map((match, index) => (
                <div key={index} className="match-card">
                  <div className="match-header">
                    <div>
                      <div className="match-title">{match.name}</div>
                      <div className="match-brand">{match.brand}</div>
                    </div>
                  </div>
                  <div className="match-meta">
                    <span>{match.gender || 'Unisex'}</span>
                    {match.rating && (
                      <span className="rating-badge">★ {match.rating.toFixed(1)}/5</span>
                    )}
                  </div>
                  <div className="notes-section">
                    {match.top_notes && <div><strong>Top:</strong> {match.top_notes}</div>}
                    {match.middle_notes && <div><strong>Middle:</strong> {match.middle_notes}</div>}
                    {match.base_notes && <div><strong>Base:</strong> {match.base_notes}</div>}
                  </div>
                  {match.main_accords && (
                    <div className="accords-tags">
                      {match.main_accords.split(',').map((acc, i) => (
                        <span key={i} className="tag">{acc.trim()}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
