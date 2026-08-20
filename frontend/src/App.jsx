import "./App.css";
import { useRef, useState } from "react";

function App() {
  // =========================================
  // CHATBOT OPEN / CLOSE
  // =========================================

  const [isOpen, setIsOpen] = useState(true);

  // =========================================
  // CHAT MESSAGES
  // =========================================

  const [messages, setMessages] = useState([]);

  // =========================================
  // LOADING
  // =========================================

  const [isLoading, setIsLoading] = useState(false);

  // =========================================
  // USER INPUT
  // =========================================

  const [inputMessage, setInputMessage] = useState("");

  // =========================================
  // SPEECH STATE
  // =========================================

  const [isListening, setIsListening] = useState(false);

  const [isHoldingMic, setIsHoldingMic] = useState(false);

  const recognitionRef = useRef(null);

  const finalTranscriptRef = useRef("");

  // =========================================
  // QUICK QUESTIONS
  // =========================================

  const suggestions = [
    "I have fever and cough",
    "I have a headache",
    "I have stomach discomfort",
    "Help me choose a product",
  ];

  // =========================================
  // SEND MESSAGE
  // =========================================

  const sendMessage = async (messageText) => {
    const message = messageText.trim();

    if (!message || isLoading) {
      return;
    }

    // Stop speech if still active
    stopListening();

    // Add user message
    setMessages((previousMessages) => [
      ...previousMessages,
      {
        sender: "user",
        text: message,
      },
    ]);

    // Clear input
    setInputMessage("");

    // Start loading
    setIsLoading(true);

    try {
      // =====================================
      // SEND TO FASTAPI
      // =====================================

      const response = await fetch(
        "http://127.0.0.1:8000/chat",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            message: message,
          }),
        }
      );

      // =====================================
      // CHECK RESPONSE
      // =====================================

      if (!response.ok) {
        throw new Error(
          `Backend returned ${response.status}`
        );
      }

      // =====================================
      // READ RESPONSE
      // =====================================

      const data = await response.json();

      console.log("SITA response:", data);

      // =====================================
      // ADD BOT RESPONSE
      // =====================================

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          sender: "bot",
          text:
            data.response ||
            data.answer ||
            "Sorry, I couldn't generate a response.",
        },
      ]);
    } catch (error) {
      console.error(
        "Frontend → Backend error:",
        error
      );

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          sender: "bot",
          text:
            "Sorry, I couldn't connect to SITA right now. Please make sure the backend server is running.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // =========================================
  // SEND BUTTON
  // =========================================

  const handleSend = () => {
    sendMessage(inputMessage);
  };

  // =========================================
  // QUICK QUESTION
  // =========================================

  const handleSuggestionClick = (suggestion) => {
    sendMessage(suggestion);
  };

  // =========================================
  // ENTER KEY
  // =========================================

  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSend();
    }
  };

  // =========================================
  // START SPEECH RECOGNITION
  // =========================================

  const startListening = () => {
    // Don't start while chatbot is processing
    if (isLoading) {
      return;
    }

    // Prevent duplicate recognition
    if (recognitionRef.current) {
      return;
    }

    const SpeechRecognition =
      window.SpeechRecognition ||
      window.webkitSpeechRecognition;

    // =====================================
    // BROWSER SUPPORT
    // =====================================

    if (!SpeechRecognition) {
      alert(
        "Speech recognition is not supported in this browser. Please use Google Chrome."
      );

      return;
    }

    // =====================================
    // CREATE RECOGNITION
    // =====================================

    const recognition = new SpeechRecognition();

    // Indian English
    recognition.lang = "en-IN";

    // Keep listening while button is held
    recognition.continuous = true;

    // Show speech while user is speaking
    recognition.interimResults = true;

    recognition.maxAlternatives = 1;

    // Reset previous transcript
    finalTranscriptRef.current = "";

    // =====================================
    // RECOGNITION STARTED
    // =====================================

    recognition.onstart = () => {
      console.log("🎤 SITA is listening...");

      setIsListening(true);
    };

    // =====================================
    // RECOGNITION RESULT
    // =====================================

    recognition.onresult = (event) => {
      let interimTranscript = "";

      for (
        let i = event.resultIndex;
        i < event.results.length;
        i++
      ) {
        const transcript =
          event.results[i][0].transcript;

        if (event.results[i].isFinal) {
          finalTranscriptRef.current +=
            transcript + " ";
        } else {
          interimTranscript += transcript;
        }
      }

      const completeTranscript = (
        finalTranscriptRef.current +
        interimTranscript
      ).trim();

      // Put speech into chatbot input
      setInputMessage(completeTranscript);
    };

    // =====================================
    // RECOGNITION ERROR
    // =====================================

    recognition.onerror = (event) => {
      console.error(
        "Speech recognition error:",
        event.error
      );

      setIsListening(false);
      recognitionRef.current = null;
    };

    // =====================================
    // RECOGNITION ENDED
    // =====================================

    recognition.onend = () => {
      console.log(
        "🎤 Speech recognition stopped."
      );

      setIsListening(false);
      recognitionRef.current = null;
    };

    // Store recognition
    recognitionRef.current = recognition;

    // =====================================
    // START
    // =====================================

    try {
      recognition.start();
    } catch (error) {
      console.error(
        "Could not start speech recognition:",
        error
      );

      recognitionRef.current = null;
      setIsListening(false);
    }
  };

  // =========================================
  // STOP SPEECH RECOGNITION
  // =========================================

  const stopListening = () => {
    if (recognitionRef.current) {
      console.log(
        "🛑 Stopping speech recognition..."
      );

      try {
        recognitionRef.current.stop();
      } catch (error) {
        console.log(
          "Speech recognition already stopped."
        );
      }

      recognitionRef.current = null;
    }

    setIsListening(false);
  };

  // =========================================
  // MICROPHONE PRESS START
  // =========================================

  const handleMicPressStart = (event) => {
    event.preventDefault();

    if (isLoading) {
      return;
    }

    setIsHoldingMic(true);

    startListening();
  };

  // =========================================
  // MICROPHONE PRESS END
  // =========================================

  const handleMicPressEnd = (event) => {
    event.preventDefault();

    setIsHoldingMic(false);

    stopListening();
  };

  // =========================================
  // OPEN CHAT
  // =========================================

  const openChat = () => {
    setIsOpen(true);
  };

  // =========================================
  // CLOSE CHAT
  // =========================================

  const closeChat = () => {
    setIsHoldingMic(false);

    stopListening();

    setIsOpen(false);
  };

  // =========================================
  // UI
  // =========================================

  return (
    <>
      {isOpen && (
        <div className="app">

          {/* ===================================
              HEADER
          =================================== */}

          <header className="header">

            <div className="brand">

              <div className="sita-logo-circle">
                🩺
              </div>

              <div className="brand-text">

                <h1 className="sita-logo">
                  SITA
                </h1>

                <p>
                  Health Assistant
                </p>

              </div>

            </div>

            <button
              type="button"
              className="close-button"
              onClick={closeChat}
              aria-label="Close SITA"
            >
              ×
            </button>

          </header>


          {/* ===================================
              CHAT AREA
          =================================== */}

          <main className="chat-area">

            {/* =================================
                WELCOME
            ================================= */}

            {messages.length === 0 && (
              <>
                <section className="welcome">

                  <h2>
                    Hello
                  </h2>

                  <p>
                    How can I help you today?
                  </p>

                </section>


                {/* QUICK QUESTIONS */}

                <div className="suggestions">

                  {suggestions.map(
                    (suggestion, index) => (
                      <button
                        key={index}
                        type="button"
                        className="suggestion-button"
                        onClick={() =>
                          handleSuggestionClick(
                            suggestion
                          )
                        }
                      >
                        {suggestion}
                      </button>
                    )
                  )}

                </div>
              </>
            )}


            {/* =================================
                MESSAGES
            ================================= */}

            {messages.length > 0 && (
              <div className="messages">

                {messages.map(
                  (message, index) => (
                    <div
                      key={index}
                      className={
                        message.sender === "user"
                          ? "message user-message"
                          : "message bot-message"
                      }
                    >
                      {message.text}
                    </div>
                  )
                )}


                {/* LOADING */}

                {isLoading && (
                  <div className="message bot-message loading-message">
                    SITA is thinking...
                  </div>
                )}

              </div>
            )}

          </main>


          {/* ===================================
              INPUT AREA
          =================================== */}

          <section className="input-area">

            {/* =================================
                HOLD TO SPEAK MESSAGE
            ================================= */}

            {isHoldingMic && (
              <div className="speech-hint">
                🎙️ Hold to speak…
              </div>
            )}


            <div className="input-wrapper">

              {/* =================================
                  INPUT
              ================================= */}

              <input
                type="text"
                className="chat-input"
                placeholder={
                  isListening
                    ? "Listening..."
                    : "Ask SITA..."
                }
                value={inputMessage}
                onChange={(event) =>
                  setInputMessage(
                    event.target.value
                  )
                }
                onKeyDown={handleKeyDown}
                disabled={isLoading}
              />


              {/* =================================
                  PUSH-TO-TALK MICROPHONE
                  
                  HOLD = LISTEN
                  RELEASE = STOP
              ================================= */}

              <button
                type="button"
                className={`mic-button ${isListening
                    ? "listening"
                    : ""
                  } ${isHoldingMic
                    ? "mic-button-active"
                    : ""
                  }`}
                onPointerDown={
                  handleMicPressStart
                }
                onPointerUp={
                  handleMicPressEnd
                }
                onPointerCancel={
                  handleMicPressEnd
                }
                onPointerLeave={(event) => {
                  if (isHoldingMic) {
                    handleMicPressEnd(event);
                  }
                }}
                onContextMenu={(event) => {
                  event.preventDefault();
                }}
                disabled={isLoading}
                aria-label="Hold to speak"
                title="Hold to speak"
              >
                {isListening
                  ? "🎙️"
                  : "🎤"}
              </button>


              {/* =================================
                  SEND
              ================================= */}

              <button
                type="button"
                className="send-button"
                onClick={handleSend}
                disabled={
                  isLoading ||
                  !inputMessage.trim()
                }
                aria-label="Send message"
              >
                →
              </button>

            </div>


            {/* =================================
                DISCLAIMER
            ================================= */}

            <p className="disclaimer">
              For general health and wellness
              guidance.
            </p>

          </section>

        </div>
      )}


      {/* =========================================
          CHAT LAUNCHER
      ========================================= */}

      {!isOpen && (
        <button
          type="button"
          className="chat-launcher"
          onClick={openChat}
          aria-label="Open SITA Health Assistant"
        >

          <span className="launcher-icon">
            🩺
          </span>

          <span className="launcher-pulse"></span>

        </button>
      )}
    </>
  );
}

export default App;