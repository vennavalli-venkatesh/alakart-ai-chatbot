const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

export const isSpeechRecognitionSupported = () => {
    return !!SpeechRecognition;
};

export const createSpeechRecognizer = ({
    onResult,
    onStart,
    onEnd,
    onError,
}) => {
    if (!SpeechRecognition) {
        return null;
    }

    const recognition = new SpeechRecognition();

    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
        onStart?.();
    };

    recognition.onresult = (event) => {
        let transcript = "";

        for (
            let i = event.resultIndex;
            i < event.results.length;
            i++
        ) {
            transcript += event.results[i][0].transcript;
        }

        onResult?.(
            transcript,
            event.results[event.results.length - 1].isFinal
        );
    };

    recognition.onend = () => {
        onEnd?.();
    };

    recognition.onerror = (event) => {
        onError?.(event.error);
    };

    return recognition;
};